# Shap-E

## 😅 로컬 테스트의 어려움.

M1 Mac은 사용할수 없습니다. CPU가 아닌 GPU를 사용해야합니다. CPU는 모델 만들때 너무 느리거든요. 만약 M1을 사용하기 위해서 "mps"로 변경 한들

```
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device("mps")
```
설정을 변경해도 Shap-e 소스 안에서 여러가지 지원하지 않기 때문에 사용할 수 없습니다.
---
## 😊 로컬 테스트 방법

Google Colab를 통해 가상의 GPU로 테스트 완료.

---
## 😅 GPU 서버를 구해야하는 어려움

cpu 무료 배포 플랫폼은 많지만
gpu 지원 서버들은 비싸구나.

google cloud platform 생각중

GPU 서버를 구해야함...
