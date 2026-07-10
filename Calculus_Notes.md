# Calculus for Data Science — Worked Notes
### eMobilis · Data Science & AI

Same structure as the other notes: concept → formula → hand-worked example, no code.

Calculus answers two questions: **how fast is something changing right now?** (derivative) and **how much has accumulated?** (integral). A matatu's speedometer is a derivative (rate of change of position); its odometer is an integral (accumulated distance). In ML, "learning" means reducing error, and reducing error means following a rate of change downhill — which is why calculus is the engine behind almost every model that learns.

---

## 1. Derivatives — the Rate of Change

For a straight line, slope is constant: $\text{slope}=\Delta y/\Delta x$. Most real functions curve, so the slope changes from point to point. The **derivative** gives the exact slope at a single point — the instantaneous rate of change.

$$f'(x) = \lim_{h\to0}\frac{f(x+h)-f(x)}{h}$$

In practice we use rules instead of the limit directly — most importantly the **power rule**: $\dfrac{d}{dx}x^n = nx^{n-1}$, constants disappear ($\frac{d}{dx}c=0$), and derivatives of sums add term-by-term.

**Worked example:** a matatu's distance from the stage after $t$ hours is $s(t)=t^2+3t$.
$$s'(t) = 2t+3$$
At $t=1$: $s'(1) = 2(1)+3 = 5$ km/h — the instantaneous speed at exactly the 1-hour mark.

**Why it matters in ML:** read the derivative as **sensitivity** — nudge the input a little, how much does the output move? A model's loss function has a derivative that tells you which way is "downhill" (less error) — that's gradient descent (Section 6).

---

## 2. The Chain Rule — Functions Inside Functions

Picture three meshed gears: A turns B at 2×, B turns C at 3× — so A turns C at 2×3=6×. Rates along a chain **multiply**.

$$\frac{dy}{dx} = \frac{dy}{du}\cdot\frac{du}{dx}$$

**Worked example:** $y=(3x+1)^2$. Let $u=3x+1$, so $y=u^2$.
$$\frac{dy}{du}=2u \qquad \frac{du}{dx}=3$$
$$\frac{dy}{dx} = 2u\times3 = 6(3x+1)$$
At $x=1$: $u=3(1)+1=4$, so $\dfrac{dy}{dx}=6(4)=24$.

**Why it matters in ML:** a neural network is one giant chain: input → layer1 → layer2 → … → loss. The chain rule computes how the loss changes with a weight buried deep inside, by multiplying rates backward through every layer. That's **backpropagation** (Section 7).

---

## 3. Integrals — Accumulation and Area

If a derivative measures rate of change, an **integral** measures how much has piled up — the area under the rate curve.

**Indefinite integral** (the antiderivative) loses the original constant, so it carries a "+C":
$$\int(2x+3)\,dx = x^2+3x+C$$

**Definite integral** has limits — the constant cancels, leaving one number (an area).

**Worked example:** speed $v(t)=2t+3$ from $t=0$ to $t=2$ hours. Antiderivative: $F(t)=t^2+3t$.
$$\int_0^2(2t+3)\,dt = F(2)-F(0) = (4+6)-(0+0) = 10$$
The matatu travelled exactly **10 km** over those 2 hours.

**Why it matters in ML:** integrals underlie probability — the area under a density curve is a probability, and the total area under any valid density is 1 (this is exactly the PDF/CDF relationship from the stats notes). Means, variances, and Bayesian normalising constants are all integrals.

---

## 4. The Fundamental Theorem of Calculus — the Bridge

**Position** is the odometer reading (total accumulated so far); **speed** is how fast it climbs (the rate). Speed is the derivative of position — and, going the other way, position is the integral of speed.

$$\int_a^b f'(x)\,dx = f(b)-f(a)$$

**Worked example:** using $s(t)=t^2+3t$ from Section 1. $s(0)=0$, $s(2)=4+6=10$. Net change $=10-0=10$ km — exactly matching the definite integral of the speed function computed in Section 3. Two completely different routes (differentiating position, or integrating speed) land on the same number — differentiation and integration are inverse operations.

---

## 5. Multivariate Calculus — Partial Derivatives and the Gradient

Real outcomes depend on many inputs at once — a model might have thousands of weights, each a dial affecting the output. A **partial derivative** measures the effect of turning *one* dial while holding all others fixed.

**Worked example:** $f(x,y)=x^2+y^2$ (a bowl-shaped surface).
$$\frac{\partial f}{\partial x}=2x \qquad \frac{\partial f}{\partial y}=2y$$

At the point $(3,4)$: $\partial f/\partial x = 6$, $\partial f/\partial y = 8$.

**Worked example 2 — the cake:** quality $Q(t,T)=-0.1t^2+5t-0.2T^2+8T-100$, where $t$=time, $T$=temperature.
$$\frac{\partial Q}{\partial T} = -0.4T+8$$
At $T=180°C$: $-0.4(180)+8 = -64$ (baking way too hot — quality is dropping fast). At $T=150°C$: $-0.4(150)+8=-52$ — still dropping, but far less steeply. The message: lower the temperature.

### The Gradient — direction of steepest ascent

The gradient stacks all partial derivatives into one vector: $\nabla f = \left(\dfrac{\partial f}{\partial x}, \dfrac{\partial f}{\partial y}\right)$. It points in the direction the function increases **fastest**. To minimise (go downhill), step in the **opposite** direction.

**Worked example:** at $(3,4)$ on $f(x,y)=x^2+y^2$:
$$\nabla f = (6,\ 8), \qquad \|\nabla f\| = \sqrt{6^2+8^2}=\sqrt{100}=10$$
This vector points straight outward (away from the bowl's centre) — the steepest uphill direction. Gradient descent walks the opposite way, $(-6,-8)$, to head toward the minimum.

---

## 6. Gradient Descent and the Cost Function

A linear model predicts $\hat y = wx+b$. We don't know the best $w,b$ — we learn them by minimising the **cost function** (mean squared error):
$$J(w,b) = \frac{1}{n}\sum_i\big(y_i-(wx_i+b)\big)^2$$

**Update rule** — feel the slope, step downhill, repeat:
$$w \leftarrow w-\alpha\frac{\partial J}{\partial w}, \qquad b\leftarrow b-\alpha\frac{\partial J}{\partial b}$$

### Warm-up: the simplest case, $J(w)=w^2$

$\dfrac{dJ}{dw}=2w$. Start at $w=3$, learning rate $\alpha=0.1$:

| Step | Gradient $2w$ | Update $w-\alpha(2w)$ | New $w$ |
|---|---|---|---|
| 1 | 6.0 | $3-0.1(6.0)$ | 2.400 |
| 2 | 4.8 | $2.4-0.1(4.8)$ | 1.920 |
| 3 | 3.84 | $1.92-0.1(3.84)$ | 1.536 |

Each step shrinks $w$ toward the minimum at $w=0$, exactly as expected — the gradient keeps pointing back toward zero and getting smaller as we approach it.

### The full model: three matatu rides

Distance ($x$) and fare ($y$): $(1,2),\ (2,4),\ (3,5)$. Gradient formulas:
$$\frac{\partial J}{\partial w} = -\frac{2}{n}\sum_i(y_i-\hat y_i)x_i \qquad \frac{\partial J}{\partial b} = -\frac{2}{n}\sum_i(y_i-\hat y_i)$$

**Starting from $w=0,\ b=0$:** predictions are $\hat y_i=0$ for all three, so errors $=y_i$: $[2,4,5]$.

$$\frac{\partial J}{\partial w} = -\frac{2}{3}\big[(2)(1)+(4)(2)+(5)(3)\big] = -\frac{2}{3}(2+8+15) = -\frac{2}{3}(25) = -16.67$$
$$\frac{\partial J}{\partial b} = -\frac{2}{3}(2+4+5) = -\frac{2}{3}(11) = -7.33$$

**One update step** with $\alpha=0.1$:
$$w \leftarrow 0-0.1(-16.67) = 1.667 \qquad b \leftarrow 0-0.1(-7.33) = 0.733$$

Already, after just one step, the line is tilting sharply toward the data. Repeating this update thousands of times converges to $w\approx1.5,\ b\approx0.667$ — **exactly the same answer** the linear algebra notes reached by solving the normal equation directly ($\hat\beta=(X^\top X)^{-1}X^\top y$). Two completely different methods, same answer — a nice sanity check that both are doing the same underlying job.

**Why calculus?** Without it, you'd have to guess-and-check random values of $w,b$. The derivative tells you exactly which direction reduces error, so every step is guaranteed progress (for a small enough $\alpha$).

---

## 7. Neural Networks and Backpropagation

A neuron: multiply each input by a weight, add a bias, pass through an activation. Learning = adjusting weights via: **forward pass** (compute output) → **measure error** → **backward pass** (how much did each weight contribute to the error — a partial derivative) → **update** (nudge weights downhill).

**Backpropagation is the chain rule, applied efficiently to every weight in the network.**

**Worked example — one neuron, one gradient step (identity activation, for simplicity):**

Input $x=2$, weight $w=0.5$, bias $b=1$, target $y=3$. Loss $L=\frac{1}{2}(\text{out}-y)^2$.

**Forward pass:**
$$\text{out} = wx+b = 0.5(2)+1 = 2.0$$
$$L = \frac{1}{2}(2.0-3)^2 = \frac{1}{2}(1) = 0.5$$

**Backward pass (chain rule):**
$$\frac{\partial L}{\partial\text{out}} = \text{out}-y = 2.0-3 = -1.0$$
$$\frac{\partial\,\text{out}}{\partial w} = x = 2 \qquad \frac{\partial\,\text{out}}{\partial b}=1$$
$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial\text{out}}\times\frac{\partial\,\text{out}}{\partial w} = (-1.0)(2) = -2.0$$
$$\frac{\partial L}{\partial b} = (-1.0)(1) = -1.0$$

**Update** with $\alpha=0.1$:
$$w \leftarrow 0.5-0.1(-2.0) = 0.7 \qquad b\leftarrow 1-0.1(-1.0) = 1.1$$

The weight and bias both moved in the direction that reduces the loss — exactly what backprop does, just for one neuron instead of millions.

---

## 8. Convolutional Neural Networks (CNNs)

A 100×100 image has 10,000 pixels — connecting every pixel to every neuron needs millions of weights. CNNs instead slide small **filters (kernels)** across the image.

**Convolution:** place the filter over a patch, multiply each filter value by the pixel underneath, sum the products into one output number, slide over, repeat.

**Worked example** — a vertical-edge filter over a dark-left/bright-right patch:
$$\text{filter} = \begin{bmatrix}1&0&-1\\1&0&-1\\1&0&-1\end{bmatrix} \qquad \text{patch} = \begin{bmatrix}0&0&1\\0&0&1\\0&0&1\end{bmatrix}$$

Element-wise multiply:
$$\begin{bmatrix}0(1)&0(0)&1(-1)\\0(1)&0(0)&1(-1)\\0(1)&0(0)&1(-1)\end{bmatrix} = \begin{bmatrix}0&0&-1\\0&0&-1\\0&0&-1\end{bmatrix}$$

Sum all 9 numbers: $-3$. A non-zero output means the filter detected its pattern (an edge) at this location; a uniform patch (all one colour) would give exactly 0.

**Where's the calculus?** Filters start random and are learned via backprop — computing $\partial\text{Error}/\partial(\text{filter value})$ for every number in the filter, the same chain-rule machinery from Section 7. Convolution itself, in its continuous form, is literally an integral: $(f*g)(t)=\int f(\tau)g(t-\tau)\,d\tau$ — "accumulate contributions as you slide," straight from Section 3.

---

## 9. Regularisation — Ridge and Lasso

A model with too many free weights can fit training data perfectly while being useless on new data — **overfitting**. The fix: add a penalty for large weights.

$$\text{Ridge (L2):}\quad J_{\text{new}} = J + \lambda\sum w^2 \qquad \text{Lasso (L1):}\quad J_{\text{new}} = J + \lambda\sum|w|$$

**Worked example:** current MSE $J=2.5$, weights $w=[3,\ -2,\ 0.5]$, penalty strength $\lambda=0.1$.

**Ridge:**
$$\sum w^2 = 3^2+(-2)^2+0.5^2 = 9+4+0.25 = 13.25$$
$$\text{penalty} = 0.1\times13.25 = 1.325 \qquad J_{\text{ridge}} = 2.5+1.325 = 3.825$$

**Lasso:**
$$\sum|w| = 3+2+0.5 = 5.5$$
$$\text{penalty} = 0.1\times5.5 = 0.55 \qquad J_{\text{lasso}} = 2.5+0.55 = 3.05$$

**Effect on the gradient:** differentiating the Ridge penalty adds a term $2\lambda w$ to every weight's update — this shrinks weights a little on every step ("weight decay"). Differentiating the Lasso penalty adds $\lambda\cdot\text{sign}(w)$ — a constant pull toward zero regardless of the weight's size, which is strong enough to push some weights **all the way to zero**, effectively deleting that feature from the model (automatic feature selection).

---

## 10. Support Vector Machines (SVM)

Given two classes of points, many lines can separate them — SVM picks the one **farthest from the nearest point of each class** (the widest margin, most robust to new data).

$$\text{margin width} = \frac{2}{\|w\|}$$

Maximising the margin is the same problem as **minimising** $\frac{1}{2}\|w\|^2$ (much easier to differentiate) subject to every point being correctly classified — a constrained optimisation solved via **Lagrange multipliers** (partial derivatives set to zero). The points that end up pinning the boundary are the **support vectors**.

**Worked example:** if $\|w\|=2$, the margin is $2/2=1$. If training finds a solution with a smaller $\|w\|=1$, the margin becomes $2/1=2$ — twice as wide. This is exactly why minimising $\|w\|^2$ (small weight vector) is equivalent to maximising the margin (more robust separation).

---

# The Whole Arc — Connecting Everything

| Concept | Plain meaning | Why calculus? |
|---|---|---|
| Derivative | How fast something changes | Sensitivity of cost to a parameter |
| Integral | A continuous sum (area) | Probabilities, expectations, convolution |
| Partial derivative | Effect of one dial, others fixed | How error changes from one weight |
| Gradient | All partials at once (points uphill) | Tells us which way is downhill |
| Gradient descent | Walk downhill to the minimum | Uses the gradient to choose each step |
| Cost function | How wrong predictions are | Differentiate it to find the minimum |
| Backpropagation | Every weight's effect, efficiently | Chain rule across the whole network |
| Convolution (CNN) | Slide a filter to extract features | The discrete form of an integral |
| Regularisation | A penalty that keeps models simple | Differentiate the penalised cost |
| SVM | The widest-margin boundary | Constrained optimisation via derivatives |

**A model learns by minimising a cost.** Minimising means finding where the slope is zero (derivatives). Getting there means following the downhill gradient. Doing that through many layers means the chain rule (backprop). The probability machinery underneath is integrals. Calculus isn't a hurdle before the real material — it **is** the mechanism of learning.

---

# Quick Reference

| Concept | Formula |
|---|---|
| Power rule | $\dfrac{d}{dx}x^n = nx^{n-1}$ |
| Chain rule | $\dfrac{dy}{dx}=\dfrac{dy}{du}\cdot\dfrac{du}{dx}$ |
| Definite integral (FTC) | $\int_a^b f'(x)\,dx = f(b)-f(a)$ |
| Partial derivative | Differentiate w.r.t. one variable, hold others fixed |
| Gradient | $\nabla f = \left(\partial f/\partial x_1,\ldots,\partial f/\partial x_n\right)$ |
| Gradient descent update | $w \leftarrow w-\alpha\,\partial J/\partial w$ |
| MSE cost | $J=\frac{1}{n}\sum(y_i-\hat y_i)^2$ |
| Backprop (chain rule) | $\dfrac{\partial L}{\partial w}=\dfrac{\partial L}{\partial\text{out}}\cdot\dfrac{\partial\,\text{out}}{\partial w}$ |
| Ridge penalty | $\lambda\sum w^2$ |
| Lasso penalty | $\lambda\sum\|w\|$ |
| SVM margin | $2/\|w\|$ |
