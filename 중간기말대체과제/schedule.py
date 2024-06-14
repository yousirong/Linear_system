import matplotlib.pyplot as plt

# 샘플 데이터 생성
sampling_steps = range(1, 17)
noise_values = [1/i for i in sampling_steps]  # 예시 노이즈 스케줄

# 막대 그래프 생성
plt.figure(figsize=(10, 6))
plt.bar(sampling_steps, noise_values, color='black')

# 축 및 제목 설정
plt.xlabel('Sampling Step')
plt.ylabel('Noise')
plt.title('Noise Schedule in Diffusion Model')

# 그래프 보여주기
plt.grid(True)
plt.show()
