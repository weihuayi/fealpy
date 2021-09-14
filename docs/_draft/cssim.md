

$$
 (\phi\frac{S_0^{n+1} - S_0^{n}}{\Delta t}, w) + (C_S\frac{p^{n+1} - p^n}{\Delta t}, w) +
(S_0^{n+1}b\frac{\nabla\cdot\boldsymbol u^{n+1} - \nabla\cdot\boldsymbol u^n}{\Delta t}, w)  
=  (f_0^{n+1}, w) - (\nabla\cdot(F_0\boldsymbol v_t^{n+1}), w), \forall w \in W 
$$

$$
(\frac{\boldsymbol \kappa^{-1}}{\lambda_1 + \lambda_0} \boldsymbol v_t^{n+1}, \boldsymbol\varphi) - 
(p^{n+1}, \nabla\cdot\boldsymbol\varphi) = 0, \forall \boldsymbol \varphi\in \boldsymbol V 
$$

$$
(\nabla\cdot \boldsymbol v_t^{n+1}, w) + (C_p\frac{p^{n+1} - p^n}{\Delta t}, w) +
(b\frac{\nabla\cdot\boldsymbol u^{n+1} - \nabla\cdot\boldsymbol u^n}{\Delta t}, w) 
=  (f_0^{n+1} + f_1^{n+1}, w), \forall w \in W
$$

$$
\left(C_\boldsymbol u\boldsymbol\varepsilon(\boldsymbol u^{n+1}), \boldsymbol\varepsilon(\boldsymbol v)\right) 
- (bp^{n+1}, \nabla\cdot\boldsymbol\phi)  = - (\boldsymbol\sigma_t, \nabla\cdot\boldsymbol\phi) + <\boldsymbol g, \boldsymbol\phi>_{\Gamma}
$$

其中饱和度方程采用分片常数离散，对应隐式迎风格式如下

$$
 (\phi\frac{S_0^{n+1} - S_0^{n}}{\Delta t}, w) + (C_S\frac{p^{n+1} - p^n}{\Delta t}, w) +
(S_0^{n+1}b\frac{\nabla\cdot\boldsymbol u^{n+1} - \nabla\cdot\boldsymbol u^n}{\Delta t}, w)  
=  (f_0^{n+1}, w) - (F_0\boldsymbol v_t^{n+1}\cdot\boldsymbol n, w), \forall w \in W 
$$
