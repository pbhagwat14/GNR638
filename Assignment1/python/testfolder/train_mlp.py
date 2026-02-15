import mytorch
import random

# -------------------------------
# Simple LINEAR dataset (not XOR)
# -------------------------------
data = []
for _ in range(20):
    x1 = random.random()
    x2 = random.random()
    y = 1 if (x1 + x2 >= 1.0) else 0
    data.append(([x1, x2], y))


# -------------------------------
# Model
# -------------------------------
l1 = mytorch.Linear(2, 4)
l2 = mytorch.Linear(4, 2)

opt = mytorch.SGD([l1.W, l1.b, l2.W, l2.b], 0.1)


# -------------------------------
# Training with GRAD DIAGNOSTICS
# -------------------------------
for epoch in range(20):
    total=0

    print("\n===== EPOCH", epoch, "=====")

    for x_val, y_val in data[:3]:   # only first 3 samples for clarity

        x = mytorch.Tensor(x_val, [1,2], False)

        # forward with saved intermediates
        h_pre = l1.forward(x)          # before relu
        h = mytorch.relu(h_pre)        # after relu
        out = l2.forward(h)            # logits
        loss = mytorch.cross_entropy(out, y_val)

        # backward
        opt.zero_grad()
        loss.backward()
        '''
        # ---- GRAD PRINTS ----
        print("\nINPUT:", x_val, "LABEL:", y_val)
        print("loss:", loss.data)
        
        print("grad logits:", out.grad)
        print("grad hidden (after relu):", h.grad)
        print("grad hidden (before relu):", h_pre.grad)
        print("grad W1:", l1.W.grad)
        print("grad W2:", l2.W.grad)
        '''
        total+=loss.data[0]
        # update
        opt.step()
    print("epoch", epoch, "avg loss:", total/len(data))    
