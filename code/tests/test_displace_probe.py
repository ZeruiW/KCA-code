"""displace is sign-sensitive: it captures motion DIRECTION, which freeze cannot.

On a task where the class is the motion direction and the spatial envelope is shared
across opposite directions (so only temporal order discriminates), reversing the
actor's trajectory (the `displace` operator) should flip the prediction to the
OPPOSITE-direction class, while freezing (removing motion) should not.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from kca.data.motion_task import MotionDirectionTask
from kca.run.train_backbone import train, accuracy
from kca import interventions as itv


def test_displace_is_sign_sensitive():
    torch.manual_seed(0)
    tr = MotionDirectionTask(n=320, seed=0)
    va = MotionDirectionTask(n=80, seed=1000)
    bb = train(tr, epochs=20, seed=0, verbose=False)
    acc = accuracy(bb.net, va, list(range(len(va))))
    assert acc > 0.85, f"direction model did not learn (acc={acc}); needs temporal order"

    OPP = MotionDirectionTask.OPP
    flip_d = flip_f = nd = nf = 0
    n = 24
    for i in range(n):
        b = va.get(i); clip = b.clip; c = clip.label
        mv = b.moving_gt[0, 0]
        r = max(range(clip.num_regions()), key=lambda r: ((clip.regions[0, 0] == r) & mv).sum().item())
        phi = bb.phi(clip, c).item()
        rev = itv.displace(clip, r, mode="reverse"); frz = itv.freeze(clip, r)
        nd += phi - bb.phi(rev, c).item(); nf += phi - bb.phi(frz, c).item()
        flip_d += bb.predicted_class(rev).item() == OPP[c.item()]
        flip_f += bb.predicted_class(frz).item() == OPP[c.item()]
    flip_d /= n; flip_f /= n; nd /= n; nf /= n
    print("OK  acc=%.2f | displace-reverse flips to OPPOSITE %.0f%% vs freeze %.0f%% | "
          "necessity displace %.2f vs freeze %.2f" % (acc, 100 * flip_d, 100 * flip_f, nd, nf))
    # displace flips direction far more than freeze; freeze (motion removal) rarely flips to opposite
    assert flip_d > 0.6, flip_d
    assert flip_d > flip_f + 0.4, (flip_d, flip_f)


if __name__ == "__main__":
    test_displace_is_sign_sensitive()
    print("\nDISPLACE SIGN-SENSITIVITY TEST PASSED")
