import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


class PeronaMalikDiffusion:
    def __init__(self, image, lambda_, sigma, stepsize, n_steps):
        self.u = image.astype(np.float64)
        self.lambda_ = self._expand_param(lambda_, n_steps)
        self.sigma = self._expand_param(sigma, n_steps)
        self.stepsize = self._expand_param(stepsize, n_steps)
        self.n_steps = n_steps

        self.history = [self.u.copy()]  
        self.time = 0.0

    def _expand_param(self, param, n):
        if np.isscalar(param):
            return np.full(n, param)
        return np.array(param)

    
    def _isodifstep(self, x, d):
        # shift (roll)
        x_up    = np.roll(x,  1, axis=0)
        x_down  = np.roll(x, -1, axis=0)
        x_left  = np.roll(x,  1, axis=1)
        x_right = np.roll(x, -1, axis=1)

        d_up    = np.roll(d,  1, axis=0)
        d_down  = np.roll(d, -1, axis=0)
        d_left  = np.roll(d,  1, axis=1)
        d_right = np.roll(d, -1, axis=1)

        
        x_xpo = x - x_up
        x_xmo = x_down - x
        x_xop = x - x_left
        x_xom = x_right - x

        
        d_dpo = d + d_up
        d_dmo = d + d_down
        d_dop = d + d_left
        d_dom = d + d_right

        
        y = 0.5 * (
            (d_dmo) * (x_xmo)
            - (d_dpo) * (x_xpo)
            + (d_dom) * (x_xom)
            - (d_dop) * (x_xop)
        )

        return y

    def compute_edges(self, image=None):
        """
        Calcola la mappa dei bordi (magnitudine del gradiente)
        """
        if image is None:
            image = self.u

        gx = np.gradient(image, axis=0)
        gy = np.gradient(image, axis=1)

        edges = np.sqrt(gx**2 + gy**2)
        return edges
    
    def plot_edges(self, step_idx=None):
        """
        Visualizza i bordi per uno stato specifico
        """
        if step_idx is None:
            image = self.u
            title = "Edges (final)"
        else:
            image = self.history[step_idx]
            title = f"Edges (step {step_idx})"

        edges = self.compute_edges(image)

        plt.imshow(edges, cmap='gray')
        plt.title(title)
        plt.axis('off')
        plt.show()
    
    def step(self, i):
      
        u_smooth = gaussian_filter(self.u, self.sigma[i])

        
        gradx = np.gradient(u_smooth, axis=0)
        grady = np.gradient(u_smooth, axis=1)
        grad2 = gradx**2 + grady**2

        
        g = 1.0 / (1.0 + grad2 / (self.lambda_[i]**2))

        
        du = self._isodifstep(self.u, g)
        self.u = self.u + self.stepsize[i] * du

        self.time += self.stepsize[i]
        self.history.append(self.u.copy())

    def run(self, verbose=False, plot_every=1):
        for i in range(self.n_steps):
            self.step(i)

            if verbose and (i % plot_every == 0):
                self.plot(i)

        return self.u

    def plot(self, step_idx):
        plt.imshow(self.u, cmap='gray')
        plt.title(f"Step {step_idx}")
        plt.axis('off')
        plt.show()

    def plot_evolution(self, ncols=5):
        n = len(self.history)
        nrows = int(np.ceil(n / ncols))

        plt.figure(figsize=(15, 3 * nrows))
        for i, img in enumerate(self.history):
            edges = self.compute_edges(img)
            plt.subplot(nrows, ncols, i + 1)
            plt.imshow(edges, cmap='gray')
            plt.title(f"{i}")
            plt.axis('off')
        plt.tight_layout()
        plt.show()