 ## Salary Prediction using machine learning

 _**hypothesis**_

 <img src="https://render.githubusercontent.com/render/math?math=h_{\Theta}(x) = \Theta^{T}x">

 _**Cost function**_

 <img src="https://render.githubusercontent.com/render/math?math=J(\Theta) = \frac{1}{2m} \sum_{i=1}^n (h_{\Theta}(x^{i}) - y^{i})^{2}">

 _**Gradient Descent**_

 __repeat until convergence / num_iterations__ 

 <img src="https://render.githubusercontent.com/render/math?math=\Theta_{0} := \Theta_{0} - \frac{1}{m} (h_{\theta}(x^{i}) - y^{i})">

 <img src="https://render.githubusercontent.com/render/math?math=\Theta_{1} := \Theta_{1} - \frac{1}{m} (h_{\theta}(x^{i}) - y^{i})x^{i}_{j}">