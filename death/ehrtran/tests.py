
def weight_gradient_test():
    a = torch.Tensor([1, 2, 3, 4])
    b = torch.Tensor([2, 3, 4, 5])

    w = torch.Tensor([2])
    w.requires_grad = True
    wa = torch.sum(a * w)

    grad = torch.autograd.grad(wa, w, retain_graph=True, only_inputs=True)
    print(grad)
    print(w)
    print(w.grad)

    wb = torch.sum(b * w)

    grad = torch.autograd.grad(wb, w, retain_graph=True, only_inputs=True)
    print(grad)
    print(w)
    print(w.grad)

    print("Done")


def gradient_test():
    a = torch.Tensor([1, 2, 3, 4])
    a.requires_grad = True
    b = a * 4
    c = torch.sum(b * 8)

    c.backward()
    print(a.grad)
    print(b.grad)

    # the question is whether the gradient accumulates

    # original
    a2 = torch.Tensor([1, 2, 3, 4])
    a2.requires_grad = True
    b2 = a2 * 4
    d2 = b2 * 12
    d2 = torch.sum(d2 * 2)
    d2.backward()

    print(a2.grad)
    print(b2.grad)

    # stacked
    a3 = a.detach()

    print(a.grad)
    print(a3.grad)
    b = a3 * 4
    d = b * 12
    d = torch.sum(d * 2)
    d.backward()

    # stack properly
    print(a.grad)
    print(b.grad)


def multi_pass_test():
    # test the .after_embedding() behavior during multiple passes
    class testModule(nn.Module):
        def __init__(self):
            super(testModule, self).__init__()
            self.weight = torch.nn.Parameter(torch.Tensor([1]))

        def forward(self, input):
            return self.weight * input

    input1 = torch.Tensor([1, 2, 3])
    input2 = torch.Tensor([5, 6, 7])
    t1 = testModule()
    t2 = testModule()
    t3 = testModule()

    l1 = t1(input1).sum()
    l1.backward()
    print(t1.weight.grad)

    l2 = t2(input2).sum()
    l2.backward()
    print(t2.weight.grad)

    l3_1 = t3(input1).sum()
    l3_2 = t3(input2).sum()
    l3 = l3_1 + l3_2
    l3.backward()
    print(t3.weight.grad)


def gradient_accumulation_test():
    """
    see if parameters behave as expected.
    :return:
    """
    a = torch.Tensor([1, 2, 3, 4])
    a.requires_grad = True
    b = a * 4
    c = torch.sum(b * 8)

    grad = torch.autograd.grad(c, a, retain_graph=True, only_inputs=True)
    print(grad)
    print(a.grad)
