import torch
import torch.nn.functional as F
from typing import Optional


def simclr_distill_loss_func(
        p1: torch.Tensor,
        p2: torch.Tensor,
        z1: torch.Tensor,
        z2: torch.Tensor,
        temperature: float = 0.1,
) -> torch.Tensor:
    device = z1.device

    b = z1.size(0)

    p = F.normalize(torch.cat([p1, p2]), dim=-1)
    z = F.normalize(torch.cat([z1, z2]), dim=-1)

    logits = torch.einsum("if, jf -> ij", p, z) / temperature
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()

    # positive mask are matches i, j (i from aug1, j from aug2), where i == j and matches j, i
    pos_mask = torch.zeros((2 * b, 2 * b), dtype=torch.bool, device=device)
    pos_mask.fill_diagonal_(True)

    # all matches excluding the main diagonal
    logit_mask = torch.ones_like(pos_mask, device=device)
    logit_mask.fill_diagonal_(True)
    logit_mask[:, b:].fill_diagonal_(True)
    logit_mask[b:, :].fill_diagonal_(True)

    exp_logits = torch.exp(logits) * logit_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positives
    mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_mask.sum(1)
    # loss
    loss = -mean_log_prob_pos.mean()
    return loss


def get_positive_logits(z1, z2):
    device = z1.device
    b = z1.size(0)
    z = torch.cat((z1, z2), dim=0)
    z = F.normalize(z, dim=-1)
    logits = torch.einsum("if, jf -> ij", z, z) / 0.1
    pos_mask = torch.zeros((2 * b, 2 * b), dtype=torch.bool, device=device)
    pos_mask[:, b:].fill_diagonal_(True)
    pos_mask[b:, :].fill_diagonal_(True)
    logit_mask = torch.ones_like(pos_mask, device=device).fill_diagonal_(0)
    logits = torch.exp(logits)
    logits = logits / (logits * logit_mask).sum(1, keepdim=True)
    pos_logits = logits * pos_mask
    return pos_logits.sum(1)

def simclr_loss_func(
        z1: torch.Tensor,
        z2: torch.Tensor,
        temperature: float = 0.1,
        extra_pos_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Computes SimCLR's loss given batch of projected features z1 from view 1 and
    projected features z2 from view 2.

    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
        z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
        temperature (float): temperature factor for the loss. Defaults to 0.1.
        extra_pos_mask (Optional[torch.Tensor]): boolean mask containing extra positives other
            than normal across-view positives. Defaults to None.

    Returns:
        torch.Tensor: SimCLR loss.
    """

    device = z1.device

    b = z1.size(0)
    z = torch.cat((z1, z2), dim=0)
    z = F.normalize(z, dim=-1)

    logits = torch.einsum("if, jf -> ij", z, z) / temperature
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    # logits = logits - logits_max.detach()
    print('logits:', logits)
    print('logits.shape:', logits.shape)

    # positive mask are matches i, j (i from aug1, j from aug2), where i == j and matches j, i
    pos_mask = torch.zeros((2 * b, 2 * b), dtype=torch.bool, device=device)
    pos_mask[:, b:].fill_diagonal_(True)
    pos_mask[b:, :].fill_diagonal_(True)
    print('pos_mask', pos_mask)

    # if we have extra "positives"
    if extra_pos_mask is not None:
        pos_mask = torch.bitwise_or(pos_mask, extra_pos_mask)

    # all matches excluding the main diagonal
    logit_mask = torch.ones_like(pos_mask, device=device).fill_diagonal_(0)
    print('logit_mask', logit_mask)
    exp_logits = torch.exp(logits) * logit_mask
    print('exp_logits', exp_logits)
    logits = torch.exp(logits)
    logits = logits / exp_logits.sum(1, keepdim=True)
    logits = logits * pos_mask
    print('logits:', logits)
    print('logits_sum', logits.sum(1))
    print('pos_logits',get_positive_logits(z1,z2))



    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
    # print('log_prob',log_prob)

    # compute mean of log-likelihood over positives
    mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_mask.sum(1)
    # loss
    loss = -mean_log_prob_pos.mean()
    return loss


if __name__ == '__main__':
    # a1 = torch.rand([2, 5])
    # a2 = torch.rand([2, 5])
    # b1 = torch.rand([2, 5])
    # b2 = torch.rand([2, 5])
    # c = simclr_loss_func(a1, a2)
    # print(c)
    a1 = torch.rand([2, 5])
    a2 = torch.rand([2, 5])
    b1 = torch.rand([2, 5])
    b2 = torch.rand([2, 5])
    A = get_positive_logits(a1, a2)
    B=get_positive_logits(b1,b2)
    print(A)
    print(B)
    c1=A[:2]>A[:2]
    print(c1)
    c2=A[2:4]>=B[2:4]
    print(c2)
    print(c1|c2)
    print(a1)
    print(a1[c1|c2])
    print(a1[False,False])
    print(sum(c2))
    a=torch.tensor([[True,True],[True,True]])
    b = torch.tensor([False,False])
    print(a)
    a[range(len(a)), range(len(a))] = b
    print(a)


