'use strict';
const equations={
  anchor:String.raw`\mathcal L_{\mathrm{anchor}}=\frac12\lVert U_k-\bar U_k\rVert_F^2`,
  hoyer:String.raw`\begin{aligned}\mathcal L_{\mathrm{Hoyer}}={}&\frac{1}{|\mathcal A|}\sum_{i=1}^{|\mathcal A|}\frac{\lVert u_{i,:}\rVert_1}{\lVert u_{i,:}\rVert_2+\epsilon}\\&+\frac{1}{|\mathcal Z|}\sum_{j=1}^{|\mathcal Z|}\frac{\lVert u_{:,j}\rVert_1}{\lVert u_{:,j}\rVert_2+\epsilon}\\&+\frac12\lVert U^\top U-I\rVert_F^2\end{aligned}`,
  objective:String.raw`\begin{aligned}\mathcal L_{\mathrm{actor}}={}&\mathcal L_{\mathrm{SAC}}\\&+\lambda_a\mathcal L_{\mathrm{anchor}}\\&+\lambda_h\mathcal L_{\mathrm{Hoyer}}\end{aligned}`,
  decoder:String.raw`a=\tanh\!\left(\omega\left(\tilde U z+b\right)\right)`,
  ema:String.raw`\tilde U_k=(1-\tau)\tilde U_{k-1}+\tau U_k`,
  sparsity:String.raw`H(v)=\frac{\sqrt n-\lVert v\rVert_1/\lVert v\rVert_2}{\sqrt n-1}`
};
function drawMath(element,latex){if(element&&window.katex)katex.render(latex,element,{displayMode:true,throwOnError:true,output:'htmlAndMathml',trust:false,strict:'error'});}
document.querySelectorAll('[data-equation]').forEach(e=>drawMath(e,equations[e.dataset.equation]));
document.querySelectorAll('[data-inline-math]').forEach(e=>katex.render(e.dataset.inlineMath,e,{throwOnError:true,output:'htmlAndMathml',trust:false}));
window.updateDecoderMath=function(devyn){drawMath(document.getElementById('decoder-math'),devyn?equations.decoder:String.raw`a\sim\pi_\theta(\,\cdot\mid s\,)`);};
