import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from discriminator import Discriminator
from vqgan import VQGAN
from utils import DATASETS, get_spec, load_data, models_dir, weights_init, plot_comparison


def parse_args():
    p = argparse.ArgumentParser(description="Entrenamiento del VQGAN (FAWN / GIROS / CAEZ / FRAUNHOFER / 5GMMGESTURE)")
    p.add_argument("--dataset", type=str, default="GIROS", choices=DATASETS.keys())
    p.add_argument("--quantizer", type=str, default="ema", choices=["ema", "original"])
    p.add_argument("--n_embeddings", type=int, default=512, help="K: número de vectores del codebook")
    p.add_argument("--embedding_dim", type=int, default=64, help="Números por vector (igual que el VQ-VAE)")
    p.add_argument("--downsample", type=int, default=16, choices=[4, 16],
                   help="Reducción total del encoder (16 = VQGAN original, 4 = como el VQ-VAE). "
                        "En 5GMMGESTURE se reparte entre los dos ejes (4 x 4 o 2 x 2)")
    p.add_argument("--beta", type=float, default=0.25)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--n_updates", type=int, default=5000)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--beta2", type=float, default=0.9)
    p.add_argument("--disc_start", type=int, default=500,
                   help="Update a partir del cual actúa el discriminador")
    p.add_argument("--disc_factor", type=float, default=1.0,
                   help="Peso del discriminador (0 = sin discriminador, para la prueba de ablación)")
    p.add_argument("--disc_norm", type=str, default="batch", choices=["batch", "none"],
                   help="Normalización del discriminador (batch = BatchNorm, none = sin normalización)")
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--eval_interval", type=int, default=1000,
                   help="Cada cuántos updates se evalúa en validación y se guarda el modelo")
    p.add_argument("--val_batches", type=int, default=50,
                   help="Lotes de validación en cada evaluación (0 = todos)")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("-save", action="store_true")
    p.add_argument("--filename", type=str, default=None,
                   help="Ruta del .pth (por defecto: <dataset>/vqgan-models-{q}[_x4]/nembeddings{K}/model.pth)")
    args = p.parse_args()

    # Lo que necesitan el modelo y el discriminador, sacado de la tabla de datasets.
    # Todo esto se guarda en los hiperparámetros del checkpoint, y el compressor
    # reconstruye el modelo exactamente igual a partir de ellos.
    spec = get_spec(args.dataset, args.downsample)
    args.image_channels = spec["channels"]
    args.height = spec["height"]
    args.width = spec["width"]
    args.down2d = spec["down2d"]
    args.latent_dim = args.embedding_dim
    args.num_codebook_vectors = args.n_embeddings

    if args.filename is None:
        sub = f"nembeddings{args.n_embeddings}" + ("_sin_disc" if args.disc_factor == 0 else "")
        args.filename = os.path.join(models_dir(args.dataset, args.quantizer, args.downsample),
                                     sub, "model.pth")
    return args


def nmse_ratios(real, recon):
    """NMSE lineal de cada grid: ||x - x_hat||^2 / ||x||^2 (los grids con potencia 0 se descartan)."""
    mse = torch.sum((real - recon) ** 2, dim=(1, 2, 3))
    power = torch.sum(real ** 2, dim=(1, 2, 3))
    keep = power > 0
    return (mse[keep] / power[keep]).cpu()


@torch.no_grad()
def evaluate(vqgan, loader, height, width, device, max_batches=0):
    """NMSE de validación: media en LINEAL de todos los grids y conversión a dB al final."""
    vqgan.eval()
    ratios, last = [], None
    for b, (x, _) in enumerate(loader):
        if max_batches and b >= max_batches:
            break
        x = x.to(device)
        recon = vqgan.reconstruct(x)              # ya sin filas ni columnas de padding
        ratios.append(nmse_ratios(x[:, :, :height, :width], recon))
        last = (x, recon)
    vqgan.train()
    r = torch.cat(ratios)
    nmse_db = 10 * np.log10(r.mean().item()) if len(r) > 0 else float("nan")
    return nmse_db, last


def save_checkpoint(path, vqgan, discriminator, results, args):
    """Mismo formato que el VQ-VAE: model, results, hyperparameters (+ discriminador)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({
        "model": vqgan.state_dict(),
        "discriminator": discriminator.state_dict(),
        "results": results,
        "hyperparameters": dict(vars(args)),
    }, path)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True   # elige las convoluciones más rápidas para estas formas
    h, w = args.height, args.width
    carpeta = "resultados_vqgan" if args.downsample == 16 else f"resultados_vqgan_x{args.downsample}"
    plot_dir = os.path.join(carpeta, args.dataset, args.quantizer)

    print(f"--> Dispositivo: {device}")
    print(f"--> Dataset: {args.dataset} | canales: {args.image_channels} | tamaño real: {h} x {w}")
    print(f"--> Reducción x{args.downsample}" + (" repartida en los dos ejes" if args.down2d else " en el eje vertical"))
    print(f"--> Cuantizador: {args.quantizer} | K = {args.n_embeddings} | dimensión = {args.embedding_dim}")
    print(f"--> Discriminador: factor {args.disc_factor}, desde el update {args.disc_start}, norm = {args.disc_norm}")
    if args.save:
        print(f"--> Modelo: {args.filename}")

    train_loader, val_loader, _, x_train_var = load_data(args)

    vqgan = VQGAN(args).to(device)
    discriminator = Discriminator(args).to(device)
    discriminator.apply(weights_init)

    # Con EMA el codebook no lleva gradiente: solo se optimiza lo que lo tiene
    opt_vq = torch.optim.Adam(
        [p for p in vqgan.parameters() if p.requires_grad],
        lr=args.learning_rate, betas=(args.beta1, args.beta2), eps=1e-8,
    )
    opt_disc = torch.optim.Adam(
        discriminator.parameters(),
        lr=args.learning_rate, betas=(args.beta1, args.beta2), eps=1e-8,
    )

    results = {
        "n_updates": 0,
        "recon_errors": [], "loss_vals": [], "perplexities": [],
        "g_losses": [], "d_losses": [], "lambdas": [],
        "val_nmse_db": [],          # lista de (update, NMSE en dB)
    }

    steps_per_epoch = len(train_loader)
    print(f"--> {args.n_updates} updates | {steps_per_epoch} updates por época "
          f"(≈ {args.n_updates / steps_per_epoch:.2f} épocas)")

    vqgan.train()
    discriminator.train()
    train_iter = iter(train_loader)
    pbar = tqdm(range(args.n_updates), desc="Entrenando", unit="upd", dynamic_ncols=True)

    for i in pbar:
        try:
            x, _ = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, _ = next(train_iter)
        x = x.to(device)
        x_real = x[:, :, :h, :w]                  # sin filas ni columnas de padding

        # ---------------- Generador (encoder + codebook + decoder) ----------------
        decoded, _, q_loss, perplexity = vqgan(x)
        x_fake = decoded[:, :, :h, :w]

        # Misma pérdida de reconstrucción que el VQ-VAE: MSE / varianza del train
        rec_loss = torch.mean((x_fake - x_real) ** 2) / x_train_var

        disc_factor = vqgan.adopt_weight(args.disc_factor, i, threshold=args.disc_start)
        B = x_real.shape[0]
        if disc_factor > 0:
            # Real y reconstruido en el MISMO lote: con lotes separados la BatchNorm
            # deja al discriminador distinguir lotes en vez de grids (ver diagnóstico)
            logits = discriminator(torch.cat([x_real, x_fake], dim=0))
            g_loss = -torch.mean(logits[B:])
            lam = vqgan.calculate_lambda(rec_loss, g_loss)
        else:
            # Antes de disc_start no se calcula nada del discriminador (ni λ)
            g_loss = torch.zeros((), device=device)
            lam = torch.zeros((), device=device)

        loss = rec_loss + q_loss + disc_factor * lam * g_loss

        opt_vq.zero_grad()
        loss.backward()
        opt_vq.step()

        # ---------------- Discriminador (pérdida hinge) ----------------
        d_loss = torch.zeros((), device=device)
        if disc_factor > 0:
            logits = discriminator(torch.cat([x_real, x_fake.detach()], dim=0))  # detach: no toca el generador
            d_real, d_fake = logits[:B], logits[B:]
            d_loss = 0.5 * (torch.mean(F.relu(1.0 - d_real)) + torch.mean(F.relu(1.0 + d_fake)))

            opt_disc.zero_grad()
            (disc_factor * d_loss).backward()
            opt_disc.step()

        # ---------------- Registro ----------------
        results["recon_errors"].append(rec_loss.item())
        results["loss_vals"].append(loss.item())
        results["perplexities"].append(perplexity.item())
        results["g_losses"].append(g_loss.item())
        results["d_losses"].append(d_loss.item())
        results["lambdas"].append(lam.item())
        results["n_updates"] = i + 1

        if i % 10 == 0:
            pbar.set_postfix(
                epoca=f"{i / steps_per_epoch:.2f}",
                recon=f"{np.mean(results['recon_errors'][-10:]):.4f}",
                perp=f"{np.mean(results['perplexities'][-10:]):.1f}",
                D=f"{np.mean(results['d_losses'][-10:]):.3f}",
            )

        if i % args.log_interval == 0:
            n = args.log_interval
            tqdm.write(
                f"Update #{i} | Recon: {np.mean(results['recon_errors'][-n:]):.5f} | "
                f"Loss: {np.mean(results['loss_vals'][-n:]):.5f} | "
                f"Perp: {np.mean(results['perplexities'][-n:]):.1f} | "
                f"G: {np.mean(results['g_losses'][-n:]):.4f} | "
                f"D: {np.mean(results['d_losses'][-n:]):.4f} | "
                f"λ: {np.mean(results['lambdas'][-n:]):.4f}"
            )

        if (i + 1) % args.eval_interval == 0 or (i + 1) == args.n_updates:
            val_db, last = evaluate(vqgan, val_loader, h, w, device, args.val_batches)
            results["val_nmse_db"].append((i + 1, val_db))
            tqdm.write(f"[Update {i + 1}] NMSE de validación: {val_db:.2f} dB")
            if last is not None:
                plot_comparison(last[0], last[1], h, save_path=os.path.join(
                    plot_dir, f"val_K{args.n_embeddings}_upd{i + 1}.png"), width=w)
            if args.save:
                save_checkpoint(args.filename, vqgan, discriminator, results, args)

    pbar.close()
    if args.save:
        save_checkpoint(args.filename, vqgan, discriminator, results, args)
        print(f"--> Modelo final guardado en: {args.filename}")


if __name__ == "__main__":
    main()


