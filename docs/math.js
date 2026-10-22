'use strict';
const equations={
  anchor:String.raw`\begin{aligned}\mathcal L_{\mathrm{anchor}}={}&\lambda_U\,\lVert U-\bar U\rVert_F^2\\&+\lambda_b\,\lVert b-\bar b\rVert_2^2\end{aligned}`,
  hoyer:String.raw`\begin{aligned}\mathcal L_{\mathrm{Hoyer}}={}&\lambda_c\sum_j\frac{\lVert U_{:j}\rVert_1}{\sqrt{\lVert U_{:j}\rVert_2^2+\varepsilon}}\\&+\lambda_r\sum_i\frac{\lVert U_{i:}\rVert_1}{\sqrt{\lVert U_{i:}\rVert_2^2+\varepsilon}}\end{aligned}`,
  decoder:String.raw`a=\tanh\!\left(\tau\left(U_{\mathrm{EMA}}z+b_{\mathrm{EMA}}\right)\right)`,
  ema:String.raw`U_{\mathrm{EMA}}\leftarrow(1-\alpha)U_{\mathrm{EMA}}+\alpha U`,
  sparsity:String.raw`H(v)=\frac{\sqrt n-\lVert v\rVert_1/\lVert v\rVert_2}{\sqrt n-1}`
};
function drawMath(element,latex){if(element&&window.katex)katex.render(latex,element,{displayMode:true,throwOnError:true,output:'htmlAndMathml',trust:false,strict:'error'});}
document.querySelectorAll('[data-equation]').forEach(e=>drawMath(e,equations[e.dataset.equation]));
document.querySelectorAll('[data-inline-math]').forEach(e=>katex.render(e.dataset.inlineMath,e,{throwOnError:true,output:'htmlAndMathml',trust:false}));
window.updateDecoderMath=function(devyn){drawMath(document.getElementById('decoder-math'),devyn?equations.decoder:String.raw`a\sim\pi_\theta(\,\cdot\mid s\,)`);};
