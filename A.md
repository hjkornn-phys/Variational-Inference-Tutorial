# 들어가며
이 시리즈는 Variational Inference를 소개, 실습하고 더 나아가 VAE로 손글씨를 생성해보는 과정을 거칠 예정입니다. 이번 포스트에서는 분포 추정을 하는 이유에 대해 살펴보겠습니다.
-출처가 표기되지 않은 모든 그림은 original content 입니다.
# 의문 - 분포를 왜 추정하는걸까?
![img.svg](https://upload.wikimedia.org/wikipedia/commons/f/f3/Beta_distribution_pdf.svg)
출처: https://upload.wikimedia.org/wikipedia/commons/f/f3/Beta_distribution_pdf.svg
## warm-up
Parameter의 값을 추정하는 방법은 대표적으로 MLE(Maximum Likelihood Estimation)와 MAP(Maximum A Posterior)가 있습니다. 두 가지 방법으로 최적의 parameter를 추정할 수 있다면 parameter의 분포에 대해 꼭 알아야 할까요? MLE와 MAP의 차이점에 대해 간단히 살펴보겠습니다. 
### 1. MLE
MLE는 관측치 $X=\{x_1, x_2, ..., x_n\}$ 이 주어졌을 때, 관측치 $$x_i$$가 등장할 확률의 곱인 likelihood $p(X|\theta)=\prod_i p(x_i|\theta)$ 를 사용합니다. Likelihood를 parameter $\theta$ 에 대해 최대화하면 얻은 관측치나 데이터 포인트를 가장 잘 설명하는 parameter $\theta_{MLE}$ 를 구할 수 있습니다. 

MLE의 문제점은 일반화 성능에서 나타납니다. 주사위를 던질 때, 각 면이 등장할 확률을 $\theta_i$라고 한다면 그 확률 분포는 다음과 같습니다. 
$$
p(x;\theta_1, \theta_2, ..., \theta_6)=\prod_i \theta_i^{\textbf{I}(x=i)}
$$
이 때, 관측치 $X=[1,3,6,5,2,1]$ 에 대한 likelihood는 $\theta_1^2\theta_3\theta_6\theta_5\theta_2$가 됩니다. 4의 눈은 등장하지 않았는데, $\sum \theta_i =1$ 라는 조건을 만족하도록 likelihood를 최대화한다면 $\theta_4=0$ 을 얻습니다. 4의 눈이 등장할 확률이 0이라는 결론은 주어진 데이터를 잘 설명하지만, 주사위를 6번 던지고 4의 눈이 나오지 않았다고 그런 일은 일어나지 않는다고 단언할 수 없습니다. **한 번도 보지 않은 데이터**에 대해 0을 부여하는 MLE 방식에는 문제가 있어 보입니다.

**결론적으로, MLE는 주어진 데이터를 잘 설명하지만, 데이터 분포에 대한 가설을 활용하지 않습니다.**


### 2. MAP
MLE가 주어진 데이터를 바탕으로 추론한다면, MAP는 가설을 활용합니다. 가설은 "데이터는 이러한 분포일거야." 혹은 "데이터는 $$Beta$$ 분포를 따르고 parameter는 $\alpha=5, \beta=4$ 일거야."등의 주장이나 믿음입니다. 이미 알고 있는 정보나 가설을  이용하여 설정한 parameter의 분포가 prior $p(\theta)$ 입니다. 
Bayes 정리에 의해, posterior $p(\theta|X)$ 는 다음과 같이 주어집니다.  
$$
p(\theta|X) = \frac{p(X|\theta)p(\theta)}{p(X)}
$$
Posterior에는 MLE에서 살펴본 확률의 곱 $p(X|\theta)$ 와 $p(\theta)$ 가 곱해져 있습니다. 즉, prior가 likelihood의 가중치로 사용됩니다. Posterior를 $\theta$에 대해 최대화한다면 $\theta_{MAP}$ 를 얻을 수 있습니다. 
$$
\theta_{MAP} = \argmax{\prod_i p(x_i|\theta)p(\theta)}
$$
동전을 10번 던져서 앞면이 7번 나왔을 때, MLE를 사용한 사람은 "이 동전의 앞면이 나올 확률 ($\theta_{MLE}$)은 0.7이다."라고 생각합니다. MAP를 사용하기 위해 동전이 앞면이 나올 확률에 대한 가설을 세워 보겠습니다. prior은 임의로 설정하였고, likelihood는 $_{10}C_7\;\theta^7(1-\theta)^3$ 으로 계산하였습니다.

| 가설 | prior | likelihood |
|:---:|:---:|:---:|
| 0.5 | 0.8 | 0.1172 |
| 0.6 | 0.2 | 0.2150 |

posterior는 prior와 likelihood의 곱에 비례하므로, 그 값을 최대로 만드는 parameter가 $\theta_{MAP}$입니다. 

| 가설 | prior | likelihood | prior * likelihood | posterior |
|:---:|:---:|:---:|:---:|:---:|
| 0.5 | 0.8 | 0.1172 | 0.09376 | 0.6855 |
| 0.6 | 0.2 | 0.2150 | 0.043 | 0.3144 |

따라서 $\theta_{MAP}=0.5$ 이고, $\theta_{MAP}\neq\theta_{MLE}$를 확인했습니다. MLE와는 다르게, MAP는 기존의 가설(prior)와 데이터의 영향(likelihood)을 고려하여 parameter를 추정합니다.

**결론적으로, MAP는 가설을 활용한다고 할 수 있습니다.**

### 3. MLE와 MAP의 관계
위의 예시에서 MAP를 사용하기 위해 갑자기 prior를 설정했습니다. Prior 없이 MAP를 사용할 수 있을까요?  
$$
\theta_{MAP} = \arg\max{\prod_i p(x_i|\theta)p(\theta)}
$$
MAP에서 사용한 식에서, $p(\theta)$ 가 상수, 즉 uniform 분포라면 MLE에서 사용한 식 $\theta_{MLE} = \arg\max\prod_i p(x_i|\theta)$ 과 같아지게 됩니다. 따라서 MLE는 uniform prior를 갖는 MAP의 특수한 경우입니다.

# 처음으로 돌아가서...
지금까지 MLE와 MAP로 parameter를 추정해 보았습니다. 이제 처음의 질문 "분포 추정은 왜 하는걸까?"로 돌아가겠습니다. 이 질문은 "분포 추정이 꼭 필요할까?" 와도 연결되어 있는데요. 먼저 MLE와 MAP만으로 해결할 수 없는 문제를 살펴보겠습니다.

## 문제: 럭키 슬롯 머신은 어느 쪽?
출처: https://towardsdatascience.com/mle-map-and-bayesian-inference-3407b2d6d4d9

카지노에 승률 50%로 설정된 슬롯 머신이 잔뜩 있습니다. 들리는 말에 의하면 이 카지노에 승률 67%인 **럭키 슬롯 머신**이 한 대 있다는데, **럭키 슬롯 머신**을 찾아서 잭팟을 터트리고 싶습니다. 눈여겨 볼 만한 머신은 딱 두 대 A, B인데, 지켜본 결과 머신 A에서는 4번 중 3번을 이겼고, 머신 B에서는 121번 중 81번 이겼습니다. 과연 A와 B 중 어느 쪽이 **럭키 슬롯 머신**일까요?

### 1. MAP로 추정하기
MAP 방법을 사용하려면, 적절한 prior를 정해야 합니다. 이항 분포와 관련되고(conjugate prior) 0.5에서 최댓값을 갖는 $$Beta(2,2)$$를 prior로 설정하겠습니다. 이 prior는  "나는 저 머신의 승률이 0.5일 확률이 가장 크다고 생각해"라는 가설을 표현합니다.
머신의 승률 prior를 $$Beta$$ 분포로 설정하면 분포의 최빈값(mode)을 parameter에 대한 식으로 나타낼 수 있고, $$Beta$$ 분포에 이항 분포를 곱한 분포 또한 $$Beta$$ 분포의 형태가 되기 때문에, parameter 업데이트 규칙을 안다면 쉽게 MAP를 적용할 수 있습니다.

```python
from scipy.stats import beta
import matplotlib.pyplot as plt
import numpy as np


fig, ax = plt.subplots(1, 1)
a, b = 2,2 
x = np.linspace(beta.ppf(0.001, a, b),
                beta.ppf(0.999, a, b), 1000)
beta_2_2 = beta.pdf(x,a,b)
ax.plot(x, beta_2_2,
       'r-', lw=5, alpha=0.6, label='beta pdf')
ax.set_xticks([0,0.2,0.4,0.6,0.8,1.0])
ax.set_xlim(0,1)
ax.set_xlabel("win_probability(p)")
ax.axvline((a-1)/(a+b-2), color = 'k')
ax.text((a-1)/(a+b-2) * 1.05 , 1, f'mode = {(a-1)/(a+b-2)}')
plt.title("Prior: Beta(2,2)")
plt.show()
```
![](https://images.velog.io/images/gibonki77/post/f9500237-4dd9-4730-8996-4a12e7d475a1/image.png)

Prior로 쓰인 $$Beta(\alpha, \beta)$$의 최빈값(mode)은 $$\frac{\alpha-1}{\alpha+\beta-2}$$로 주어지고, 관측치에 승리가 $$k$$번, 패배가 $$(n-k)$$번 이라면 posterior는 $$Beta(\alpha+k, \beta+n-k)$$가 된다고 알려져 있습니다. 따라서 posterior를 최대로 하는 승률 $$P_{MAP}$$는 posterior의 mode입니다.
$$
P_{MAP} = \frac{\alpha+k-1}{\alpha+\beta+n-2}$$

머신 A와 B의 승률을 MAP로 계산해 보겠습니다. A에서 4번 중 3번 승리, B에서 121번 중 81번 승리를 관찰했습니다.
$$
P_{A, MAP} = \frac{2+3-1}{2+2+4-2} = \frac{2}{3}
\\
P_{B, MAP} = \frac{2+81-1}{2+2+121-2} = \frac{2}{3}
\\
\therefore P_{A, MAP} = P_{B, MAP}
$$
A의 관찰 횟수보다 B의 관찰 횟수가 많은데 MAP로 추정된 승률은 A와 B가 같습니다.
그렇다면 A가 **럭키 슬롯머신**일 확률과 B가 **럭키 슬롯머신**일 확률이 동등하다고 할 수 있을까요?
그렇지 않습니다. MAP로는 posterior의 최빈값(mode)만 알 수 있을 뿐, 그 추정이 어느 정도의 확신도를 갖는지 알 수 없습니다.

### 2. Posterior 확인하기
이제 머신 A, B의 승률 posterior $$Beta(5, 3), Beta(83, 42)$$를 살펴보겠습니다. 
```python
fig, ax = plt.subplots(1, 1)
a, b = 5,3 
x = np.linspace(beta.ppf(0.001, a, b),
                beta.ppf(0.999, a, b), 10000)

beta_5_3 = beta.pdf(x,5,3)
beta_83_42 = beta.pdf(x,83,42)

ax.plot(x, beta_5_3,
       'r-', lw=5, alpha=0.6, label='Pdf_A: Beta(5,3) ')
ax.plot(x, beta_83_42,
       'b-', lw=5, alpha=0.6, label='Pdf_B: Beta(83, 42)')
ax.set_xticks([0,0.2,0.4,0.6,0.8,1.0])
ax.set_xlim(0,1)
ax.set_xlabel("win_probability(p)")
ax.axvline((a-1)/(a+b-2), color = 'k')
ax.text((a-1)/(a+b-2) * 1.05 , 3.5, f'mode = {(a-1)/(a+b-2):.3f}')
plt.legend()
plt.title("Posterior")
plt.show()
```
![](https://images.velog.io/images/gibonki77/post/80314692-1642-4dd6-8977-4f9aacb68c2d/image.png)

머신 A, B의 posterior를 시각화하면, 위의 그림과 같습니다. (A: 빨강, B: 파랑)
두 분포의 최빈값은 0.66으로 같지만 머신 B의 승률 분포가 최빈값 근처에 몰려 있습니다. 머신 B의 신뢰 구간이 머신 A보다 좁기 때문에 머신 B를 **럭키 슬롯 머신**으로 추정하는 것이 타당합니다.  또한 이 선택은 승률이 0.5일 때 4번 중 3번 승리는 충분히 일어날 수 있지만, 121번 중 81번 승리는 잘 일어나지 않는다는 직관과 부합합니다. 
이처럼 MLE(uniform prior를 갖는 posterior의 최빈값), MAP(posterior의 최빈값) 뿐만 아니라 posterior를 알아야 해결할 수 있는 문제가 존재합니다. **사실, 분포 추정은 Machine Learning의 핵심 사항으로 분포를 완벽히 알게 된다면 다른 많은 문제도 해결됩니다.**

# 분포 추정의 이유
Machine Learning에서 추정하고 싶은 분포 $p(x_i)$ 는 test set의 데이터 분포로, test set에서 데이터 $x_i$를 보게 될 확률을 표현합니다. 분포 $p(x_i)$ 를 알게 된다면 다음과 같은 작업을 할 수 있습니다.
* 이상치 탐지: $p(x_i)$ 가 작은 값을 갖는 $$x_i$이상치로 골라 낼 수 있습니다.
* 결측치 채우기: 결측치는 $p(x_i)$ 를 기반으로 채울 수 있습니다.
* 벡터 양자화: $p(x_i)$ 가 큰 값을 갖는 $x_i$ 로 test set을 군집화할 수 있습니다.

분포 $p(x_i, y_i)$ 마저 알게 된다면 추가 작업이 가능합니다.
* 지도 학습: $p(y_i|x_i) = \frac{p(x_i, y_i)}{p(x_i)}$ 이므로, 데이터 $x_i$의 label을 $\argmax_y p(y_i|x_i)$ 로 구할 수 있습니다.

**결론적으로, 분포를 정확하게 추정하면 Machine Learning의 많은 문제를 해결할 수 있어, 빠르고 정확한 추정을 위한 많은 추정 기법이 발달했습니다.**


# 생성모델과 판별모델
Machine Learning에서 분류 모델을 문제 해결 방식에 따라 크게 두 가지, 생성모델과 판별모델로 나눌 수 있습니다. 샘플 데이터셋을 $X_{sample}$, 그에 따른 label set을 $Y_{sample}$라고 할 때, 생성모델은 $X_{sample}$가 $p(X)$로부터 생성되었다고 가정하고 분포 $p(X), p(X,Y)$ 를 추정하여 posterior $p(Y|X)$ 를 얻는 반면, 판별모델은 $p(Y|X)$ 를 바로 추정합니다. 다르게 말한다면, 생성모델은 **데이터의 분포를 학습하여 생성 규칙**을 파악하고, 판별모델은 **클래스 간 차이에 집중해 decision boundary**를 학습합니다.

예를 들어, CNN(LeNet)을 사용한 **냥멍 모델**에 개와 고양이 데이터셋을 입력으로 개-고양이 분류 문제를 해결하는 경우를 생각해 보겠습니다. 샘플 $x_i$는 각 layer를 연달아 통과하고, 모델은 최종 결과와 label $y_i$의 binary cross entropy loss를 최소화하도록 weight를 학습합니다. 그 결과 분류를 잘하는 모델을 얻었습니다. 

**냥멍 모델**은 판별모델입니다. 모든 이미지의 집합 $S$ 에 모델의 분류 결과 $p(Y=cat|X=x_{boundary}) \approx p(Y=dog|X=x_{boundary})$ 가 되는 경계점 $x_{boundary}$ 이 존재하고, 경계점은 모여서 경계를 만듭니다. $S$ 에서 개와 고양이 데이터셋이 어떤 분포를 갖는지, 개 이미지는 어떻게 분포되어 있는지 관심을 두지 않고, 적절한 decision boundary를 얻었습니다. **냥멍 모델**로 그럴듯한 개 이미지를 생성할 수 있을까요?

그렇지 않습니다. **냥멍 모델** 기준으로 $p(Y=dog|X)$가 1에 가까운 이미지를 모아 놓더라도 그 집합 안에는 실제 개와 전혀 관계없는 이미지도 섞여 있기 때문입니다. 

![](https://images.velog.io/images/gibonki77/post/bbcec09a-1df1-4680-b30c-842011351380/502929_1_En_1_Fig3_HTML.png)
출처: Explainable AI with Python(Leonida Gianfagna, Antonio Di Cecco) - Springer

<br>

생성모델은 분포 $p(X), p(X,Y)$ 를 추정하는 만큼 더 복잡한 작업을 처리할 수 있지만(참고: [왜 분포를 추정하는걸까?](https://velog.io/@gibonki77/Inference-1)), 판별 모델에 비해 계산이 복잡하다는 단점이 있습니다. 
Bayes 정리에 의해 다음이 성립하고, posterior(좌변)을 얻기 위해 Y에 대한 적분(분모)이 필요하기 때문으로, 모델 구조가 복잡하고 Y가 고차원일수록 적분은 다루기 힘들어져(intractable) 그대로 계산할 수 없고 다른 방법이 필요하게 됩니다. 그 방법 중 하나인 variational inference를 살펴보겠습니다. 
$$
p(Y|X) = \frac{p(X,Y)}{p(X)} = \frac{p(X|Y)p(Y)}{p(X)} = \frac{p(X|Y)p(Y)}{\int p(X|Y) p(Y)dY}
$$

# Variational Inference

>Variational Inference라는 이름은 물리학의 variational method에서 유래했습니다. Variational method를 사용하면 실제의 경로는 action functional의 정류점에 존재한다는 원리(Stationary Action Principle)로부터 운동 방정식(Euler–Lagrange equation)을 유도할 수 있습니다.

Variational Inference는 복잡한 문제를 간단한 문제로 변화시켜 함수를 근사합니다. 이 과정에서 variational parameter라는 변수를 추가로 도입하고, 추정 문제를 최적화 문제로 변화시킵니다. 예시와 함께 살펴보겠습니다.

## $\log(x)$를 직선으로 근사하기 (Convex duality)

기울기 $\lambda$가 주어질 때, concave function $g(x) = \log(x)$를 직선으로 근사한다면 직선은 $f(x)=\lambda x - b(\lambda)$의 꼴로 나타나고 최적의 $b$를 구하는 문제가 됩니다. x를 변화시켜 가며 $\lambda x$와 $\log(x)$의 차 $\lambda x - \log(x)$를 최소화하면 그 값이 주어진 $\lambda$에 대한 최적의 $b$가 됩니다. $\lambda$에 대해 최적의 $b$를 반환하는 함수를 $f^*(\lambda)$라고 한다면, 다음의 관계를 만족합니다.
$$
f^*(\lambda) = \min_x  \{\lambda x - f(x)\}
$$
위 관계를 직선의 방정식에 대입하면 기울기가 $\lambda$이고 $\log(x)$에 접하는 직선의 모임을 얻습니다.  
$$
J(x, \lambda) = \lambda x  - f^*(\lambda) \ge g(x),\;for\;all\; \lambda, x
$$
이 중에서 $x_0$ 근처에서 잘 근사하는 직선을 찾기 위해서는 $f(x_0) = g(x_0)$이어야 하고, 등호를 만족하는 조건(J의 극소점)이 되므로, 임의의 $x$를 기준으로 $g(x)$를 잘 근사한 함수 $f(x)$를 다음과 같이 표현할 수 있습니다.
$$
f(x) = \min_\lambda \{J\} = \min_\lambda \{\lambda x - f^*(\lambda)\}
$$

variational parameter $\lambda$를 도입하여, $x$에 대해 비선형인 함수 $\log(x)$를  선형으로 근사하는 규칙을 얻었습니다. 이때 $\log(x)$의 복잡성은 $f^*(\lambda)$로 흡수되었습니다. 

## 분포로 확장하기
분포 $p(X)$에 대해서도 $A(X, \lambda) \le p(X),\;for\;all\;\lambda,X$인  $A$를 찾을 수 있다면, $p(X)$를 근사한 함수 $q(X)$를 다음과 같이 정할 수 있습니다.
$$
q(X) = \max_\lambda \{A\}
$$
하지만 $p(X)$를 알지 못하는 상황에서 lower bound를 구하려면 어떻게 해야 하는지 전혀 모르겠습니다. 어떻게 해야 할까요?

## Variational Inference 의 수식 유도

수식의 자세한 유도보다는 그 의미를 따라가 보겠습니다. $p(X)$는 **확률분포**이고, $q(Z|\lambda)$는 $p(Z)$에 대한 근사로, 마음대로 정할 수 있고, 미지입니다. 
* 은닉 변수 $Z$가 존재하는 모델에서 데이터의 분포 $p(X) = \sum_Z p(X,Z)$입니다. 
* 양변에 $\log$를 씌우면 Jensen 부등식을 통해 lower bound를 표현할 수 있습니다.
*  $q(Z|\lambda)$에서 $\lambda$는  variational parameter이고,  $\lambda$가 $q$의 parameter로 작동한다는 표현입니다.
* $KL(p||q) = \sum_Z p(Z) \log \frac{p(Z)}{q(Z)}$ 로 정의되고, 분포 간 "얼마나 떨어져 있는지" 표현하는 척도입니다.

$$
\begin{aligned}
\log p(X) &= \log(\sum_Z p(X,Z)) \\&= \log(\sum_Z p(X,Z)\;\; \frac{q(Z|\lambda)}{q(Z|\lambda)})\\&=\log(\sum_Z \;q(Z|\lambda) \;\; \frac{p(X,Z)}{q(Z|\lambda)})\\&\ge\sum_Z q(Z|\lambda) \log(\frac{p(X,Z)}{q(Z|\lambda)})
\\
\\\sum_Z [q(Z|\lambda)\log p(X,Z)\; -\;q(Z|\lambda)\log q(Z|\lambda)]&=\sum_Z [q(Z|\lambda)\log (p(X|Z) p(Z))\; -\;q(Z|\lambda)\log q(Z|\lambda)]\\&=\sum_Z [q(Z|\lambda)\log p(X|Z)\; -\;q(Z|\lambda)\log \frac{q(Z|\lambda)}{p(Z)}]
\\&= \mathbb E_{q(Z|\lambda)}[\log p(X|Z)]\; - \; KL(q(Z|\lambda)||p(Z))
\\\\
\therefore \log p(X) &\ge \mathbb E_{q(Z|\lambda)}[\log p(X|Z)]\; - \; KL(q(Z|\lambda)||p(Z))
    
\end{aligned}

$$

위 과정을 통해, $\log p(X)$에 대해 lower  bound를 얻어냈습니다. $p(X)$를 evidence라고 부르기도 하여, 우변은 **ELBO**(Evidence Lower BOund)로 부릅니다. 선형 근사의 예에서 보듯, $\lambda$를 변화시켜 주어진 데이터셋 $X$에 대해 bound를 원래의 분포에 가장 가깝게 만들 때, 가장 근사를 잘 한 경우가 됩니다. 이 때 근사의 한계는 설정한 $q$의 모양에 따라 결정됩니다. 선형 근사에서 **가장 잘 근사하는 직선**을 얻었듯이, $q$를 Gaussian으로 설정했다면 $p(X)$를 **가장 잘 근사하는 Gaussian**을 얻습니다.

따라서 $q$의 모양과 $\lambda$를 조절하여 $\mathbb{E}_{q(Z|\lambda)}[\log p(X|Z)]\; - \; KL(q(Z|\lambda)||p(Z))$를 최대화한다면 $\log p(X)$를 잘 근사했다는 결론을 얻습니다. 

추가적으로, 분포 추정 문제가 최적화 문제로 바뀌었으므로 Variational Inference는 optimization을 통해 추정, 즉 inference를 실행한다고 볼 수 있습니다.

# EM 알고리즘

## 마치며
다음 포스트에서는 분포를 추정하는 기법 중 Variational Inference에 대해 본격적으로 소개하겠습니다.

## Reference
[Beta distribution-Wikipedia](https://en.wikipedia.org/wiki/Beta_distribution)
[MLE, MAP and Bayesian Inference](https://towardsdatascience.com/mle-map-and-bayesian-inference-3407b2d6d4d9)
[CPSP 540 - Machine Learning: UBC graduate lecture](https://www.cs.ubc.ca/~schmidtm/Courses/540-W20/)