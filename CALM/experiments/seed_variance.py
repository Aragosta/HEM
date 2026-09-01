import sys; sys.path.insert(0,'/home/user/HEM'); sys.path.insert(0,'/home/user/HEM/CALM')
sys.path.insert(0,'/home/user/HEM/CALM/experiments')
import torch, statistics
from test_hierarchy import HierarchicalLanguage
from helm_calm import HelmCALM, PatchAutoencoder
from stage1_energy_head import EuclideanBackbone, CalmHead, energy_score
from tests._config import tiny_args
STEPS, LR = 4000, 1e-3
lang = HierarchicalLanguage(3, 4)
a = tiny_args(vocab_size=lang.vocab_size, dim=33, n_layers=3, max_seq_len=32, original_seq_len=32)
batches = [lang.sample(2, 32, seed=i) for i in range(16)]
def make_ae(seed=0):
    torch.manual_seed(seed)
    ae = PatchAutoencoder(lang.vocab_size, hidden=128, latent_size=32, patch_size=1)
    opt = torch.optim.AdamW(ae.parameters(), lr=3e-3)
    flat = torch.cat([b.reshape(-1) for b in batches]).view(-1,1)
    g = torch.Generator().manual_seed(seed)
    for _ in range(800):
        r = torch.randint(0, flat.size(0), (128,), generator=g)
        l,_ = ae.elbo(flat[r]); opt.zero_grad(); l.backward(); opt.step()
    return ae.freeze()
def helm(ae, seed):
    torch.manual_seed(seed)
    m = HelmCALM(a, ae, num_samples=8, head_kind="lorentz")
    g = m.parameter_groups(); ps = g["euclidean"]+g["manifold"]
    o = torch.optim.AdamW(ps, lr=LR); m.train()
    for s in range(STEPS):
        l = m.loss(batches[s%16]); o.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(ps,1.0); o.step(); m.retract_manifold_parameters()
    m.eval(); c=n=0
    with torch.no_grad():
        for t in batches:
            p,tg = m.predict_tokens(t, n_samples=32); c += (p==tg).sum().item(); n += tg.numel()
    return c/n
def euc(ae, seed, width=33):
    torch.manual_seed(seed)
    bb = EuclideanBackbone(a.vocab_size, width, a.n_layers, a.n_heads, a.inter_dim)
    hd = CalmHead(width, ae.latent_size); ps = list(bb.parameters())+list(hd.parameters())
    o = torch.optim.AdamW(ps, lr=LR); bb.train(); hd.train()
    for s in range(STEPS):
        t = batches[s%16]; tg = t[:,1:].reshape(-1)
        with torch.no_grad(): mu, ls = ae.encode(tg.unsqueeze(-1))
        h = bb(t)[:,:-1].reshape(-1,width)
        l = -energy_score(hd.sample(h.unsqueeze(0).expand(8,-1,-1)), mu, ls).mean()
        o.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(ps,1.0); o.step()
    bb.eval(); hd.eval(); c=n=0
    with torch.no_grad():
        for t in batches:
            tg = t[:,1:].reshape(-1); h = bb(t)[:,:-1].reshape(-1,width)
            d = ae.decode(hd.sample(h.unsqueeze(0).expand(32,-1,-1))).argmax(-1).squeeze(-1)
            c += (torch.mode(d,dim=0).values==tg).sum().item(); n += tg.numel()
    return c/n
ae = make_ae(); SEEDS=[0,1,2,3,4]
print(f"5 seeds, tree language, K=1, {STEPS} steps\n")
H=[helm(ae,s) for s in SEEDS]; E=[euc(ae,s) for s in SEEDS]
print(f"{'':28s} " + " ".join(f"s{s}" .rjust(7) for s in SEEDS) + f"{'mean':>9s}{'sd':>8s}")
print(f"{'CALM+HELM (Lorentz head)':28s} " + " ".join(f"{v:6.2%}" for v in H) + f" {statistics.mean(H):8.2%}{statistics.stdev(H):7.2%}")
print(f"{'CALM+Euclidean (control)':28s} " + " ".join(f"{v:6.2%}" for v in E) + f" {statistics.mean(E):8.2%}{statistics.stdev(E):7.2%}")
d = statistics.mean(E)-statistics.mean(H)
pooled = (statistics.stdev(H)**2/len(H) + statistics.stdev(E)**2/len(E))**0.5
print(f"\ndifference {d:+.2%}   standard error {pooled:.2%}   ratio {abs(d)/pooled if pooled else 0:.1f}")
