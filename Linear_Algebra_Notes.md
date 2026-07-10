# Linear Algebra for Data Science — Worked Notes
### eMobilis · Data Science & AI

Same structure as the stats notes: concept → formula → hand-worked example, no code. Every example uses small numbers you can check yourself.

---

## Module 0 — Why We Need Linear Algebra

Statistics tools (distributions, t-tests, covariance) handle **one or two variables at a time**. A real matatu fleet dataset might have 50 features across 200 matatus — 10,000 numbers. Computing Pearson r for every pair of 50 features needs 1,225 separate calculations.

**Linear algebra handles all variables simultaneously** — one matrix operation replaces thousands of one-at-a-time calculations.

| Stats concept | Linear algebra version |
|---|---|
| One matatu's data row | A **vector** |
| Your whole fleet CSV | A **matrix** |
| Cov(X,Y) | One cell in a **covariance matrix** |
| Pearson r | **Cosine similarity** of two vectors |
| Regression line | Solving **AX = B** |
| PCA directions | **Eigenvectors** of the covariance matrix |

---

## Module 1 — Vectors

**A vector is an ordered list of numbers** — one row of your dataset. **Dimension** = how many numbers it contains (just count them).

### Addition — combine component-by-component

$$\mathbf{u}+\mathbf{v} = [u_1+v_1,\ u_2+v_2,\ \ldots]$$

**Worked example:** morning shift [6 trips, 9k fare] + afternoon shift [4 trips, 7k fare]:
$$[6,9] + [4,7] = [6+4,\ 9+7] = [10,\ 16]$$

### Scalar multiplication — stretch, shrink, or reverse

**Worked example:** $v = [3,4]$
$$2v = [6,8] \quad 0.5v = [1.5,2] \quad -1v = [-3,-4]$$

### Magnitude — length of the vector (Pythagoras extended)

$$\|v\| = \sqrt{v_1^2+v_2^2+\cdots+v_n^2}$$

**Worked example:** $v=[3,4]$
$$\|v\| = \sqrt{3^2+4^2} = \sqrt{9+16} = \sqrt{25} = 5$$

### Unit vectors — pure direction, length exactly 1

$$\hat v = \frac{v}{\|v\|}$$

**Worked example:** $v=[3,4]$, length 5:
$$\hat v = \left[\frac{3}{5}, \frac{4}{5}\right] = [0.6,\ 0.8]$$
Check: $\sqrt{0.6^2+0.8^2}=\sqrt{0.36+0.64}=\sqrt{1}=1$ ✓

### The Dot Product — the single most important operation in ML

$$u\cdot v = u_1v_1+u_2v_2+\cdots+u_nv_n$$

**Worked example:** $u=[1,2,3]$, $v=[4,5,6]$
$$u\cdot v = (1\times4)+(2\times5)+(3\times6) = 4+10+18 = 32$$

**Geometric meaning:** $u\cdot v = \|u\|\|v\|\cos\theta$. Large positive = vectors point the same way; zero = perpendicular; large negative = opposite directions. A model's prediction (e.g. `LinearRegression.predict()`) is a dot product of a feature vector with a learned weight vector.

### Cosine Similarity — comparing pattern, not scale

$$\cos\_sim(u,v) = \frac{u\cdot v}{\|u\|\|v\|}$$

**Worked example:** three matatus, features = [trips, fare_KES, passengers]:
$$A=[170,\ 12000,\ 120] \qquad B=[175,\ 12500,\ 125] \qquad C=[8,\ 3000,\ 40]$$

Lengths: $\|A\|\approx12001.8$, $\|B\|\approx12501.8$, $\|C\|\approx3000.3$

$$\cos\_sim(A,B) = \frac{A\cdot B}{\|A\|\|B\|} \approx 1.000 \qquad \cos\_sim(A,C) \approx 0.9999$$

A and B (nearly identical operating patterns) score close to 1, as expected. This is **exactly Pearson r in disguise** — r is the cosine similarity of the mean-centred vectors $(X-\bar X)$ and $(Y-\bar Y)$.

### Linear Independence — does a vector add new information?

**Worked example:** $v_A=[4,8]$, $v_B=[2,4]$, $v_C=[3,7]$.

Can we build $v_B$ from $v_A$? Try $v_B = k\,v_A$: from position 0, $k=2/4=0.5$. Check position 1: $0.5\times8=4$ ✓. **Yes** — $v_B$ is a scaled copy of $v_A$; it adds no new direction. **Dependent.**

Can we build $v_C$ from $v_A$? Try $k=3/4=0.75$. Check position 1: $0.75\times8=6.0 \ne 7$. **No single scaling works** — $v_C$ points in a genuinely new direction. **Independent.**

**Why it matters:** dependent (redundant) columns in a dataset (e.g. `fare_KES` and `fare_USD`) make the matrix your model needs to invert **singular** — coefficients become meaningless or the model crashes.

---

## Module 2 — Matrices

A matrix is a rectangular grid of numbers, $m$ rows × $n$ columns. **A matrix IS your dataset** — each row an observation, each column a feature.

### Matrix Multiplication — the engine of all ML

**Rule:** element $(i,j)$ of the result = (row $i$ of A) $\cdot$ (column $j$ of B) — a dot product. For $A@B$ to work, **columns of A must equal rows of B**: $(m\times n)@(n\times p) = (m\times p)$.

**Worked example:**
$$A=\begin{bmatrix}1&2\\3&4\end{bmatrix} \qquad B=\begin{bmatrix}5&6\\7&8\end{bmatrix}$$

$$C_{00} = [1,2]\cdot[5,7] = 1(5)+2(7) = 19 \qquad C_{01} = [1,2]\cdot[6,8] = 1(6)+2(8) = 22$$
$$C_{10} = [3,4]\cdot[5,7] = 3(5)+4(7) = 43 \qquad C_{11} = [3,4]\cdot[6,8] = 3(6)+4(8) = 50$$

$$C = \begin{bmatrix}19&22\\43&50\end{bmatrix}$$

**Element-wise multiply** ($A*B$, matching positions) is a completely different operation from **matrix multiply** ($A@B$, row·column) — do not confuse them.

### Transpose — flip rows and columns

$(A^\top)_{ij} = A_{ji}$. An $m\times n$ matrix becomes $n\times m$.

**Worked example:**
$$A=\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix}_{2\times3} \qquad A^\top=\begin{bmatrix}1&4\\2&5\\3&6\end{bmatrix}_{3\times2}$$

Every ML formula with a $^\top$ is using transpose to make matrix dimensions line up for multiplication — e.g. the regression formula $\beta=(X^\top X)^{-1}X^\top y$.

### Special matrices — the Identity

The identity matrix $I$ has 1s on the diagonal, 0s elsewhere, and $A\cdot I = I\cdot A = A$ — it's the matrix version of "multiply by 1."

### The Covariance Matrix — all relationships at once

Instead of computing Cov(X,Y) one pair at a time, the covariance matrix holds variances on the diagonal and every pairwise covariance off-diagonal, all at once.

**Worked example:** trips per day (X) and revenue in hundreds (Y), reusing the stats-notes data:
$$X=[2,4,6,8,10] \qquad Y=[3,7,5,10,12]$$
From the stats notes: Var(X)=10.0, Var(Y)=13.3, Cov(X,Y)=10.5. The covariance matrix:
$$C = \begin{bmatrix}\text{Var}(X) & \text{Cov}(X,Y)\\ \text{Cov}(X,Y) & \text{Var}(Y)\end{bmatrix} = \begin{bmatrix}10.0 & 10.5\\ 10.5 & 13.3\end{bmatrix}$$

This single matrix is exactly what PCA (Module 7) decomposes.

### Rank — how much real information is in the matrix

**Rank** = number of linearly independent rows (or columns).

**Worked example:** $A = \begin{bmatrix}1&2\\2&4\end{bmatrix}$. Row 1 = 2 × Row 0 — completely redundant. **Rank = 1**, even though the matrix is 2×2. This matrix cannot be inverted (see Module 3).

---

## Module 3 — The Determinant

The determinant is a single number from a square matrix answering: **can this matrix be inverted?**

**Geometric meaning:** how much a transformation scales area (2D) or volume (3D). $\det=0$ means the transformation collapsed space onto a lower dimension — information is destroyed, no inverse exists.

**2×2 formula:**
$$\det\begin{bmatrix}a&b\\c&d\end{bmatrix} = ad-bc$$

**Worked example:**
$$A=\begin{bmatrix}7&2\\17&5\end{bmatrix}: \quad \det = (7)(5)-(2)(17) = 35-34 = 1 \quad \to \text{invertible}$$
$$B=\begin{bmatrix}2&4\\1&2\end{bmatrix}: \quad \det = (2)(2)-(4)(1) = 4-4 = 0 \quad \to \text{singular}$$

Row 1 of B = 2 × Row 0 — the rows are dependent (matches the rank-1 example above), and the determinant confirms it.

---

## Module 4 — The Matrix Inverse

$A\cdot A^{-1} = A^{-1}\cdot A = I$. If A represents a transformation, $A^{-1}$ is the exact reverse.

**2×2 formula:**
$$A=\begin{bmatrix}a&b\\c&d\end{bmatrix} \implies A^{-1} = \frac{1}{\det(A)}\begin{bmatrix}d&-b\\-c&a\end{bmatrix}$$

**Worked example**, using $A=\begin{bmatrix}7&2\\17&5\end{bmatrix}$, $\det(A)=1$:

$$A^{-1} = \frac{1}{1}\begin{bmatrix}5&-2\\-17&7\end{bmatrix} = \begin{bmatrix}5&-2\\-17&7\end{bmatrix}$$

**Check:** $A\cdot A^{-1}$ should equal $I$:
$$\begin{bmatrix}7&2\\17&5\end{bmatrix}\begin{bmatrix}5&-2\\-17&7\end{bmatrix} = \begin{bmatrix}7(5)+2(-17) & 7(-2)+2(7)\\17(5)+5(-17) & 17(-2)+5(7)\end{bmatrix} = \begin{bmatrix}1&0\\0&1\end{bmatrix} \checkmark$$

**Important rule:** never use $A^{-1}$ to solve $AX=B$ in practice — always use direct solving methods (Module 5/8); computing the inverse explicitly is slower and less numerically accurate.

---

## Module 5 — Solving Linear Systems (AX = B)

**Worked example:**
$$2x+3y=9 \qquad x-y=3$$

Matrix form: $A=\begin{bmatrix}2&3\\1&-1\end{bmatrix}$, $X=\begin{bmatrix}x\\y\end{bmatrix}$, $B=\begin{bmatrix}9\\3\end{bmatrix}$.

**Row reduction**, starting from the augmented matrix $[A\,|\,B]$:
$$\begin{bmatrix}2&3&|&9\\1&-1&|&3\end{bmatrix}$$

Step 1 — divide row 0 by 2 (leading 1):
$$\begin{bmatrix}1&1.5&|&4.5\\1&-1&|&3\end{bmatrix}$$

Step 2 — subtract row 0 from row 1 (eliminate x):
$$\begin{bmatrix}1&1.5&|&4.5\\0&-2.5&|&-1.5\end{bmatrix}$$

Read off row 1: $-2.5y=-1.5 \implies y=0.6$. Substitute into row 0: $x+1.5(0.6)=4.5 \implies x=4.5-0.9=3.6$.

**Solution:** $x=3.6,\ y=0.6$.

### Special cases

**No solution** (rank(A) < rank([A|B])): e.g. $x+y=3$ and $x+y=5$ — parallel lines that never meet; the equations contradict.

**Infinite solutions** (rank(A) = rank([A|B]) but rank < number of unknowns): e.g. $x+2y=4$ and $2x+4y=8$ — the second equation is just 2× the first; one free variable remains, giving a whole line of solutions.

---

## Module 6 — Rank, Linear Independence & the Null Space

**Null space (kernel) of A** = all vectors $x$ such that $Ax=\mathbf{0}$.

- Null space = {zero vector only} → A is full rank → inverse exists.
- Null space contains non-zero vectors → A is rank-deficient → no inverse.

**Worked example:** $A=\begin{bmatrix}1&2\\2&4\end{bmatrix}$ (rank 1, from Module 2). Solve $Ax=0$:
$$x_1+2x_2=0 \implies x_1=-2x_2$$
Any vector of the form $x=[-2k,\ k]$ satisfies this — e.g. $[-2,1]$. **The null space is the line spanned by $[-2,1]$.**

**ML intuition:** if a feature direction lands in the null space of a weight matrix, that feature has zero influence on any prediction — it's invisible to the model.

### The Four Fundamental Subspaces

| Subspace | What it is | Dimension |
|---|---|---|
| Column space | All reachable outputs $Ax$ | = rank |
| Null space | All inputs mapping to 0 | = n − rank |
| Row space | Column space of $A^\top$ | = rank |
| Left null space | Null space of $A^\top$ | = m − rank |

**Worked example**, continuing $A$ above (2×2, rank 1): column space is 1-dimensional (a line), null space is 1-dimensional ($2-1=1$, matches the line we found), row space is 1-dimensional, left null space is 1-dimensional ($2-1=1$).

---

## Module 7 — Eigenvalues and Eigenvectors

Most vectors change **both** direction and length when transformed by A. **Eigenvectors** only change length — direction stays the same (or exactly flips).

$$A v = \lambda v$$

**Worked example** — a real (non-diagonal) matrix: $A=\begin{bmatrix}4&1\\2&3\end{bmatrix}$

**Step 1 — characteristic equation:** $\det(A-\lambda I) = 0$
$$(4-\lambda)(3-\lambda) - (1)(2) = 0 \implies \lambda^2 - 7\lambda + 10 = 0 \implies (\lambda-5)(\lambda-2)=0$$
$$\lambda_1 = 5, \qquad \lambda_2 = 2$$

**Step 2 — eigenvector for $\lambda=5$:** solve $(A-5I)v=0$:
$$\begin{bmatrix}-1&1\\2&-2\end{bmatrix}v=0 \implies -v_1+v_2=0 \implies v_1=v_2$$
Eigenvector: $v=[1,1]$. Check: $Av=[4(1)+1(1),\ 2(1)+3(1)]=[5,5]=5[1,1]$ ✓

**Step 3 — eigenvector for $\lambda=2$:** solve $(A-2I)v=0$:
$$\begin{bmatrix}2&1\\2&1\end{bmatrix}v=0 \implies 2v_1+v_2=0 \implies v_2=-2v_1$$
Eigenvector: $v=[1,-2]$. Check: $Av=[4(1)+1(-2),\ 2(1)+3(-2)]=[2,-4]=2[1,-2]$ ✓

### PCA — eigenvalues applied

PCA finds the eigenvectors of the **covariance matrix**; the eigenvector with the largest eigenvalue is the direction of maximum spread (PC1).

**Worked example**, using the covariance matrix from Module 2: $C=\begin{bmatrix}10.0&10.5\\10.5&13.3\end{bmatrix}$

Solving the characteristic equation gives eigenvalues $\lambda_1 \approx 22.28$ and $\lambda_2 \approx 1.02$, with the eigenvector for $\lambda_1$ pointing in direction $\approx (0.65, 0.76)$ (normalised).

**Variance explained by PC1:**
$$\frac{22.28}{22.28+1.02} = 0.956$$

So a single direction — a combination of trips and revenue — captures **95.6%** of the total spread in this 2-feature dataset. This is exactly what "% variance explained" means in a real PCA output.

---

## Module 8 — Matrix Decompositions

A decomposition breaks a matrix into simpler factors — like $12 = 2\times2\times3$.

### LU Decomposition ($A=LU$)

L is lower-triangular, U is upper-triangular — literally Gaussian elimination stored as a matrix product.

**Worked example**, reusing $A=\begin{bmatrix}2&3\\1&-1\end{bmatrix}$ from Module 5. Eliminating row 1 using multiplier $m=1/2$ (row1 − 0.5×row0) gives:
$$U = \begin{bmatrix}2&3\\0&-2.5\end{bmatrix}, \qquad L = \begin{bmatrix}1&0\\0.5&1\end{bmatrix}$$

**Check:** $L\times U = \begin{bmatrix}1(2)+0(0) & 1(3)+0(-2.5)\\0.5(2)+1(0) & 0.5(3)+1(-2.5)\end{bmatrix} = \begin{bmatrix}2&3\\1&-1\end{bmatrix} = A$ ✓

Once L and U are known, solving $AX=B$ for many different B vectors becomes much cheaper — solve $LY=B$ then $UX=Y$, each a quick triangular substitution.

### QR Decomposition ($A=QR$)

Q has orthonormal columns (perpendicular, length 1); R is upper-triangular. Built with the **Gram-Schmidt process**.

**Worked example:** columns $a_1=[3,4]$, $a_2=[1,0]$.

Step 1 — normalise $a_1$: $q_1 = a_1/\|a_1\| = [3,4]/5 = [0.6,\ 0.8]$

Step 2 — remove the $q_1$-direction from $a_2$: projection $= (a_2\cdot q_1)q_1 = (0.6)[0.6,0.8] = [0.36,\ 0.48]$
$$a_2^{\perp} = a_2 - \text{projection} = [1,0]-[0.36,0.48] = [0.64,\ -0.48], \quad \|a_2^{\perp}\|=0.8$$
$$q_2 = a_2^{\perp}/0.8 = [0.8,\ -0.6]$$

$$R = \begin{bmatrix}\|a_1\| & a_2\cdot q_1\\ 0 & \|a_2^{\perp}\|\end{bmatrix} = \begin{bmatrix}5 & 0.6\\ 0 & 0.8\end{bmatrix}$$

$Q$'s columns ($[0.6,0.8]$ and $[0.8,-0.6]$) are perpendicular and unit-length — check: $q_1\cdot q_2 = 0.6(0.8)+0.8(-0.6)=0$ ✓. QR is the numerically stable way `lstsq`/regression solves overdetermined systems.

### SVD ($A = U\Sigma V^\top$)

Works on **any** matrix, square or not. $\Sigma$'s diagonal (the singular values) ranks directions by importance — the basis of PCA, image compression, and recommendation systems. Computing SVD by hand is impractical even for small matrices; in practice this is always done numerically, but the concept — decomposing into "important directions ranked by size" — is the same idea as eigenvalues, generalised to non-square matrices.

---

## Module 9 — How Everything Connects: Linear Regression

$$\hat\beta = (X^\top X)^{-1}X^\top y$$

**Worked example:** three matatu rides, distance (x) and fare (y): $(1,2), (2,4), (3,5)$. Add an intercept column of 1s:

$$X = \begin{bmatrix}1&1\\1&2\\1&3\end{bmatrix}, \qquad y = \begin{bmatrix}2\\4\\5\end{bmatrix}$$

**Step 1 — $X^\top X$:**
$$X^\top X = \begin{bmatrix}1&1&1\\1&2&3\end{bmatrix}\begin{bmatrix}1&1\\1&2\\1&3\end{bmatrix} = \begin{bmatrix}3&6\\6&14\end{bmatrix}$$

**Step 2 — $X^\top y$:**
$$X^\top y = \begin{bmatrix}1&1&1\\1&2&3\end{bmatrix}\begin{bmatrix}2\\4\\5\end{bmatrix} = \begin{bmatrix}2+4+5\\2+8+15\end{bmatrix} = \begin{bmatrix}11\\25\end{bmatrix}$$

**Step 3 — invert $X^\top X$ and multiply:** solving $\begin{bmatrix}3&6\\6&14\end{bmatrix}\hat\beta = \begin{bmatrix}11\\25\end{bmatrix}$ gives:
$$\hat\beta = \begin{bmatrix}0.667\\1.5\end{bmatrix}$$

So the best-fit line is $\hat y = 0.667 + 1.5x$ — intercept 0.667, slope 1.5 (i.e. each extra km of distance adds about 1.5 to the fare). This exact result reappears in the calculus notes, arrived at a completely different way — via gradient descent instead of solving the equation directly — as a nice check that both methods agree.

---

# Quick Reference

| Concept | Formula |
|---|---|
| Vector magnitude | $\|v\|=\sqrt{\sum v_i^2}$ |
| Unit vector | $\hat v = v/\|v\|$ |
| Dot product | $u\cdot v=\sum u_iv_i$ |
| Cosine similarity | $\dfrac{u\cdot v}{\|u\|\|v\|}$ |
| Matrix multiply element | $C_{ij}=$ row $i$ of A $\cdot$ col $j$ of B |
| Determinant (2×2) | $ad-bc$ |
| Inverse (2×2) | $\dfrac{1}{\det}\begin{bmatrix}d&-b\\-c&a\end{bmatrix}$ |
| Eigen-equation | $Av=\lambda v$ |
| Characteristic equation (2×2) | $\lambda^2-(\text{trace})\lambda+\det=0$ |
| Linear regression | $\hat\beta=(X^\top X)^{-1}X^\top y$ |
