# Shap-E

M1 Mac은 사용할수 없습니다. CPU가 아닌 GPU를 사용해야합니다. CPU는 모델 만들때 너무 느리거든요. 만약 M1을 사용하기 위해서 "mps"로 변경 한들

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device("mps")
설정을 변경해도 Shap-e 소스 안에서 여러가지 지원하지 않기 때문에 사용할 수 없습니다.

Google Colab를 통해 가상의 GPU로 확인 해보니 작동은 잘 됩니다.

prompt를 api로 보내고 모델을 api로 받기 개발할 예정입니다.