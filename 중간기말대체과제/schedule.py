import matplotlib.pyplot as plt

# ���� ������ ����
sampling_steps = range(1, 17)
noise_values = [1/i for i in sampling_steps]  # ���� ������ ������

# ���� �׷��� ����
plt.figure(figsize=(10, 6))
plt.bar(sampling_steps, noise_values, color='black')

# �� �� ���� ����
plt.xlabel('Sampling Step')
plt.ylabel('Noise')
plt.title('Noise Schedule in Diffusion Model')

# �׷��� �����ֱ�
plt.grid(True)
plt.show()
