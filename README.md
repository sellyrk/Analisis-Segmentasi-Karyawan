# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Jaya Jaya Maju

## Business Understanding

Perusahaan Jaya Jaya Maju adalah perusahaan yang memiliki ratusan karyawan yang aktif dengan beragam usia, latar belakang pekerjaan, pengalaman kerja, serta tingkat kepuasan kerja. Dalam menghadapi era persaingan dan tingginya tingkat turnover di industri, perusahaan merasa penting untuk memahami segmentasi atau pengelompokan sifat karyawan untuk pengambilan keputusan yang strategis.

### Permasalahan Bisnis
Dalam hal ini, perusahaan mengalami beberapa permasalahan bisnis, seperti:
1. Minimnya pemahaman tentang karakteristik setiap karyawan yang mempengaruhi dalam pengambilan keputusan kebijakan yang bersifat umum dan kurang efisien
2. Kurangnya segmentasi karyawan berdasarkan data membuat HR sulit dalam merancang strategi retensi, promosi, atau penghargaan yang efektif
3. Kesulitan dalam menemukan kelompok karyawan yang berisiko tinggi mengalami burnout atau turnover dikarenakan tidak adanya grafik atau sajian data atau analisis khusus yang menyajikan tentang informasi tersebut

### Cakupan Proyek
Proyek ini bertujuan untuk mendukung perusahaan dalam mengenali segmentasi karyawan menurut perilaku, kepuasan kerja, dan faktor-faktor penting lainnya dengan menggunakan metode clustering, dengan ruang lingkup proyek meliputi: 
1. Pemilihan fitur dari data pegawai yang tepat (usia, durasi kerja, penghasilan, tingkat kepuasan, jam lembur, dan sebagainya)
2. Penerapan metode unsupervised learning (Clustering) untuk mengelompokkan pegawai menjadi beberapa kategori
3. Visualisasi hasil pengelompokan ke dalam dashboard (grafik) untuk mempermudah pembacaan dan dijadikan dasar dalam pengambilan keputusan

### Persiapan

Sumber data: https://www.ibm.com/communities/analytics/watson-analytics-blog/watson-analytics-use-case-for-hr-retaining-valuable-employees/

#### Setup environment:

```
# Setup Env - Google Colab (Versi Python yang digunakan: 3.10+ (default Google Colab))
# Install library 
!pip install -q numpy pandas matplotlib seaborn scikit-learn joblib

# Pemanggilan Library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.stats as stats
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

```

## Business Dashboard
![selly_rk_dicoding](https://github.com/user-attachments/assets/521e94b5-24ca-405c-b384-e3cf3f0577b2)
Dari dashboard di atas, dapat dilihat jika presentase karyawan tiap keseluruhan klaster didominasi oleh klaster 2 yaitu 44.9%, dengan karyawan sebanyak 475 dari total 1058 orang. Klaster terendah dimiliki oleh klaster 1 yaitu sebesar 12.8%, dengan karyawan sebanyak 135 orang. Selain itu, distribusi departemen klaster juga dapat dilihat jika klaster 1 didominasi oleh departemen HR, sementara klaster 2 didominasi oleh departemen Sales dan klaster 0 didominasi oleh departemen HR. Dashboard juga menunjukkan jika rata-rata klaster 0 berada di 32 tahun dengan lama bekerja di perusahaan sekitar 4 tahun dan gaji paling rendah, klaster 1 rata-rata usianya 48 tahun dengan lama bekerja paling lama dan gaji paling tinggi. Klaster 2 rata-rata usianya 38 tahun dengan lama bekerja 8 tahun dan gaji sedang atau di tengah-tengah. Dari hasil ini, dapat disimpukan jika hubungan lama bekerja dan gaji yang didapat oleh karyawan adalah berbanding lurus, artinya semakin lama bekerja suatu karyawan maaka semakin tinggi gaji yang didapat. Dari seluruh klaster, klaster 2 lah yang paling sering mengambil lembur, sementara klaster 1 paling sedikit mengambl lembur. Lembur juga memengaruhi kepuasan lingkungan yang dirasakan karyawan, karyawan yang mengambil lembur cenderung lebih puas terhadap lingkungan kerjanya. Namun, ini berbanding terbalik dengan gaji yang didapat, dari dashboard dapat dilihat jika klaster karyawan yang sering mengambil lembur cenderung mendapat gaji yang lebih sedikit dari karyawan yang tidak mengambil lembur.

http://localhost:3000/public/dashboard/4bb35cd1-5a74-4762-8d7a-8294433e151f
username: sellyrizkiyah01@gmail.com password: selly140#



## Conclusion
Proyek ini berhasil membagi karyawan Perusahaan Jaya Jaya Maju menjadi beberapa segmen (cluster) berdasarkan karakteristik penting seperti usia, pendapatan, lama bekerja, tingkat kepuasan kerja, keterlibatan, dan pola lembur. Dengan menggunakan metode unsupervised learning (clustering), ditemukan pola-pola unik yang dapat dimanfaatkan perusahaan untuk sehingga menghasilkan interpretasi setiap klaster yaitu:

- Klaster 0: Karyawan entry-level/karyawan baru karena usianya yang termuda, gaji paling rendah, sering lembur, tingkat keterlibatan dan rating performa yang tinggi, dan bukan karyawan senior
- Klaster 1: Karyawan senior/berpengalaman karena usia tertua, gaji paling tinggi, paling lama tahun di perushaan, karyawan senior, jarang lembur, dan rating performa dan job involvement yang sedang
- Klaster 2: Karyawan menengah karena usia kisaran 38 tahun, gaji sedikit lebih tinggi dari klaster 0, tidak senior, lembur tidak sebanyak klaster 0 dan job involvement serta worklife balance yang sedang

### Rekomendasi Action Items (Optional)
Rekomendasi action items untuk HR/Perusahaan guna menyelesaikan permasalahan atau mencapai target mereka.
1. Cluster 0 (Entry-Level): Merekomendasikan untuk memberikan bonus tambahan untuk lembur, membuat program pengembangan karir, memberikan mentoring dari klaster 1, dan memberikan fasilitas cek kesehatan untuk mencegah level dan meningkatkan kepuasan pekerjaan.
2. Cluster 1 (Senior): Merekomendasikan untuk menjadikan mereka mentor, memberikan retention work, dan meninjau untuk meningkatkan performa dengan training refresh.
3. Cluster 2 (Mid-Level): Merekomendasikan untuk membuat program pengembangan karir dan promosi, mengajak untuk mengikuti projek leadership, menawarkan pelatihan untuk naik level dan memberikan motivasi agar selalu puas.
