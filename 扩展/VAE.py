import  torch
from    torch.utils.data import DataLoader
from    torch import nn, optim
from    torchvision import transforms, datasets
# import  visdom

class VAE(nn.Module):

    def __init__(self):
        super(VAE, self).__init__()


        # [b, 784] => [b, 20]
        # u: [b, 10]
        # sigma: [b, 10]
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 20),
            nn.ReLU()
        )
        # [b, 20] => [b, 784]
        self.decoder = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )

        self.criteon = nn.MSELoss()

    def forward(self, x):
        """
        :param x: [b, 1, 28, 28]
        :return:
        """
        batchsz = x.size(0)
        # flatten
        x = x.view(batchsz, 784)
        # encoder
        # [b, 20], including mean and sigma
        h_ = self.encoder(x)
        # [b, 20] => [b, 10] and [b, 10]
        mu, sigma = h_.chunk(2, dim=1)
        # reparametrize trick, epison~N(0, 1)
        h = mu + sigma * torch.randn_like(sigma)

        # decoder
        x_hat = self.decoder(h)
        # reshape
        x_hat = x_hat.view(batchsz, 1, 28, 28)

        kld = 0.5 * torch.sum(
            torch.pow(mu, 2) +
            torch.pow(sigma, 2) -
            torch.log(1e-8 + torch.pow(sigma, 2)) - 1
        ) / (batchsz*28*28)

        return x_hat, kld

mnist_train = datasets.MNIST('mnist', True, transform=transforms.Compose([transforms.ToTensor()]), download=False)
mnist_test = datasets.MNIST('mnist', False, transform=transforms.Compose([transforms.ToTensor()]), download=False)

mnist_train = DataLoader(mnist_train, batch_size=32, shuffle=True)
mnist_test = DataLoader(mnist_test, batch_size=32, shuffle=True)

x, _ = iter(mnist_train).next()
print('x:', x.shape)

device = torch.device('cuda')

model = VAE().to(device)
print(model)

criteon = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# viz = visdom.Visdom(use_incoming_socket=False)

for epoch in range(1000):
    for batchidx, (x, _) in enumerate(mnist_train):
        # [b, 1, 28, 28]
        x = x.to(device)
        x_hat, kld = model(x)
        loss = criteon(x_hat, x)

        if kld is not None:
            elbo = - loss - 1.0 * kld
            loss = - elbo

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(epoch, 'loss:', loss.item(), 'kld:', kld.item())

    x, _ = iter(mnist_test).next()
    x = x.to(device)
    with torch.no_grad():
        x_hat, kld = model(x)
    # viz.images(x, nrow=8, win='x', opts=dict(title='x'))
    # viz.images(x_hat, nrow=8, win='x_hat', opts=dict(title='x_hat'))
