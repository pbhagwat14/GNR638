import mytorch
import random

data = [
    ([0,0],0),
    ([0,1],1),
    ([1,0],1),
    ([1,1],0)
]

l1 = mytorch.Linear(2,8)
l2 = mytorch.Linear(8,2)

opt = mytorch.SGD([l1.W,l1.b,l2.W,l2.b],0.2)

for epoch in range(2000):

    random.shuffle(data)

    total_loss = 0

    # ---- batch gradient ----
    opt.zero_grad()

    for x_val,y_val in data:

        x = mytorch.Tensor(x_val,[1,2],False)

        h = mytorch.relu(l1.forward(x))
        out = l2.forward(h)

        loss = mytorch.cross_entropy(out,y_val)
        loss.backward()

        total_loss += loss.data[0]

    opt.step()

    if epoch%200==0:
        print("epoch",epoch,"loss",total_loss)

print("\nPredictions:")
for x_val,y_val in data:
    x = mytorch.Tensor(x_val,[1,2],False)
    h = mytorch.relu(l1.forward(x))
    out = l2.forward(h)
    print(x_val,"->",out.data)
