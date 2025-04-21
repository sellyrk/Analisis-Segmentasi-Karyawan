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
# pemrosesan data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# model dan evaluasi
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# simpan model
import joblib
```
#### Load Data
```
df = pd.read_csv("/content/employee_data.csv")
df
```
### Data Understanding
```
df.info()
```
Dari hasil informasi
Data terdiri dari 35 kolom dan 1470 baris, data berisi data demografi karyawan, yang terdiri dari kolom-kolom, yaitu:

- EmployeeId - ID Karyawan
- Attrition - Apakah terjadi pengurangan karyawan? (0 = tidak, 1 = ya)
- Age - Usia karyawan
- BusinessTravel - Keterlibatan perjalanan untuk pekerjaan
- DailyRate - Gaji harian
- Department - Departemen Karyawan
- DistanceFromHome - Jarak dari tempat kerja ke rumah (dalam km)
- Education - 1-Sekolah Menengah Pertama, 2-Sekolah Menengah Atas, 3-Sarjana, 4-Sarjana, 5-Doktor
- EducationField - Bidang Pendidikan
- EnvironmentSatisfactionn - 1-Rendah, 2-Sedang, 3-Tinggi, 4-Sangat Tinggi
- Gender - Jenis kelamin karyawan
- HourlyRate - Gaji per jam
- JobInvolvement - 1-Rendah, 2-Sedang, 3-Tinggi, 4-Sangat Tinggi
- JobLevel - Tingkat pekerjaan (1 hingga 5)
- JobRole - Peran Pekerjaan
- JobSatisfaction - 1-Rendah, 2-Sedang, 3-Tinggi, 4-Sangat Tinggi
- MaritalStatus - Status Perkawinan
- MonthlyIncome - Gaji bulanan
- MonthlyRate - Tarif per bulan
- NumCompaniesWorked - Banyaknya perusahaan tempat bekerja
- Over18 - Berusia di atas 18 tahun?
- OverTime - Lembur?
- PercentSalaryHike - Persentase kenaikan gaji tahun lalu
- PerformanceRating - 1-Rendah, 2-Baik, 3-SangatBaik, 4-LuarBiasa
- RelationshipSatisfaction - 1-Rendah, 2-Sedang, 3-Tinggi, 4-Sangat Tinggi
- StandardHours - Jam Kerja Standar
- StockOptionLevel - Tingkat Opsi Saham
- TotalWorkingYears - Lama bekerja - Total tahun bekerja
- TrainingTimesLastYear - Jumlah training yang diikuti tahun lalu
- WorkLifeBalance - 1-Rendah, 2-Baik, 3-Sangat Baik, 4-Sangat Baik
- YearsAtCompany - Lama Bekerja di Perusahaan - Tahun di Perusahaan
- YearsInCurrentRole - Lama bekerja dalam jabatan saat ini
- YearsSinceLastPromotion - Lama sejak promosi terakhir
- YearsWithCurrManager - Lama bekerja dengan manajer saat ini

### Data Preparation / Preprocessing
1. Dalam hal ini dilakukan penghapusan nilai kosong yang ada pada Attrition dengan menghapus baris-baris kosong yang ada. Setelah menghapus nilai yang kosong, dilakukan pemeriksaan data terduplikasi, namun ternyata data bebas dari duplikasi.
2. Dilakukan penambahan kolom baru yaitu ```IsSenior```, untuk mengetahui apakah karyawan tersebut adalah senior atau tidak. Dalam hal ini, senior adalah yang memiliki JobLevel di atas 3.
3. Kemudian, akan diambil beberapa fitur yang paling berpengaruh dari 35 fitur yang ada untuk klasterisasi karyawan untuk mengurangi attrition seperti: ***'EmployeeId','Age', 'Department', 'DistanceFromHome', 'EnvironmentSatisfaction', 'Gender', 'JobSatisfaction', 'OverTime', 'MonthlyIncome', 'PerformanceRating', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'WorkLifeBalance', 'JobInvolvement', 'IsSenior'***. Ini dipilih karena  merepresentasikan kondisi karyawan (usia, masa kerja, pendapatan, jarak rumah), mengukur kepuasan dan keterlibatan kerja, mengidentifikasi faktor risiko burnout atau turnover (lembur, masa jabatan saat ini, terakhir promosi), indikator kinerja dan potensi (peringkat kinerja, senioritas), dan segmentasi demografis dan struktural (departemen, gender).
4. Tahap terakhir, dilakukan encoding kategori pada fitur kategorik seperti ***'OverTime', 'IsSenior', 'Gender', dan 'Department'***

### Exploratory Data Analysis
1. Mengetahui statistik deskriptif data
Secara keseluruhan, statistik deskriptif data dari tabel di atas, rentang usia karyawan yaitu dari 18-60 tahun, rata-rata jarak rumah karyawan berada di sekitar 8.9km, gaji karyawan berkisar antara 1009 hingga 19999, dan lain-lain.
3. Memeriksa distribusi numerik
   
   ![image](https://github.com/user-attachments/assets/8906c1fc-a2f7-4ddc-95ef-896bec644c13)

Dari grafik di atas, dapat dilihat bahwa selain age, semua fitur numerik memiliki distribusi yang skewed (berbentuk right-skewed distribution)
5. Memeriksa distribusi kategorik

   ![image](https://github.com/user-attachments/assets/7a223e5d-12a6-4bfc-b037-94b4c406f762)

Dapat dilihat bahwa kebanyakan departemen berasal dari departemen 1 (Research & Development). Karyawan juga banyak yang memiliki kepuasan kerja yang sangat tinggi. Selain itu, karyawan didominasi oleh pria, dan keterlibatan pekerjaan berada di tingkat yang tinggi (3), terakhir karyawan masih banyak yang bukan senior.
7. Memeriksa fitur kategorik dan pengaruhnya pada 'Overtime'
  
   ![image](https://github.com/user-attachments/assets/d42feba4-1df9-4791-be22-945876d9995a)

Berdasarkan dari pengambilan lembur, departemen 1 (Research & Development), mengambil paling banyak dari semua karyawan, karyawan yang puas terhadap lingkungan dan pekerjaan cenderung tidak mengambil lembur, bgeitupun yang memiliki keseimbangan pola kehidupan kerja yang baik. Karyawan pria dan wanita cenderung hampir sama. Karyawan yang memiliki keterlibatan pekerjaan tinggi juga tidak mengambil lembur, dan karyawan senior juga tidak banyak yang mengambil lembur.
8. Analisis korelasi fitur numerik
  
   ![image](https://github.com/user-attachments/assets/e361ee28-bc50-4b0c-8f8a-5fce849e2afd)

Dari hasil heatmap, dapat dilihat jika fitur YearsAtCompany dan YearsInCurrentRole memiliki korelasi tertinggi sebesar 0.76, begitu juga dengan YearsAtCompany dan YearsSinceLastPromotion 0.62. Ketiga fitur ini memiliki informasi yang hampir sama. Korelasi terendah ke semua kolom adalah DistanceFromHome, sehingga kolom ini tidak begitu berpengaruh pada variabel lainnya.
9. Seleksi fitur
Hasil dari proses EDA seluruhnya, beberapa fitur perlu dihapus karena tidak berpengaruh besar, seperti Gender yang cenderung netral saat membentuk distribusi, DistanceFromHome yang berkorelasi rendah dengan variabel lain, YearsInCurrentRole dan YearsSinceLastPromotion yang redundant, sehingga hanya mempertahankan YearsAtCompany saja.
10. Standarisasi
Tahap ini dilakukan untuk menyamakan skala data supaya memudahkan dalam tahap pemodelan nantinya. Karena sebelumnya, juga terdapat kmeiirngan data ke kanan, maka data yang miring ke kanan seperti MonthlyIncome dan YearsAtCompany (keduanya numerik) akan dialkukan log transform untuk mengatasinya

### Modelling
Sebelum pemodelan, dilakukan perhitungan nilai k (jumlah klaster terbaik) dengan elbow method dan silhoutte.

![image](https://github.com/user-attachments/assets/466d1f47-af68-47c2-8c82-474de4b89536)
![image](https://github.com/user-attachments/assets/593d9db2-a799-42c1-ad73-2fe3afd6b40e)

Hasil elbow method dan silhouttte menunjukkan bahwa k terbaik berada di angka 3. Sehingga, hasil k palimg baik adalah 3 klaster. Model dibuat dengan k = 3 menggunakan KMeans dan disimpan menggunakan joblib.
```
K = 3

model = KMeans(n_clusters=K, random_state=75)
model.fit(X_scaled)

joblib.dump(model, "kmeans_clustering_model.joblib")
```
Hasil pemodelan klaster pun dibuat dengan nama 'Cluster', dimana ini adalah hasil dari klasterisasi karyawan.
```
clusters = model.predict(X_scaled)

fix_df['Cluster'] = clusters
fix_df
```

### Evaluasi
1. Visualisasi distribusi klaster

   ![image](https://github.com/user-attachments/assets/0493de9b-15a5-4cb6-b22a-19b7583e3b3c)

Terlihat jika distribusi klaster paling tinggi adalah klaster 2, jumlah klaster 2 hampir sama dengan klaster 0. Sementara, klaster terendah yaitu di cluster 1
3. Penggabungan hasil klaster dengan ```main_df``` sebelumnya untuk analisis lebih lanjut.
```result_df.groupby('Cluster').mean(numeric_only=True)```
Sehingga menghasilkan seperti ini:

|Cluster|EmployeeId|Age|Department|DistanceFromHome|EnvironmentSatisfaction|Gender|JobSatisfaction|OverTime|MonthlyIncome|PerformanceRating|YearsAtCompany|YearsInCurrentRole|YearsSinceLastPromotion|WorkLifeBalance|JobInvolvement|IsSenior|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|0|742\.3325892857143|32\.433035714285715|1\.0424107142857142|8\.732142857142858|2\.6651785714285716|0\.6183035714285714|2\.841517857142857|0\.27455357142857145|3189\.0379464285716|3\.1875|3\.533482142857143|2\.2433035714285716|0\.9955357142857143|2\.732142857142857|2\.720982142857143|0\.0|
|1|738\.7333333333333|47\.68888888888889|1\.2|7\.777777777777778|2\.8074074074074074|0\.562962962962963|2\.7111111111111112|0\.2962962962962963|17012\.792592592592|3\.111111111111111|14\.42962962962963|6\.62962962962963|4\.7481481481481485|2\.785185185185185|2\.7777777777777777|1\.0|
|2|730\.7136842105264|38\.39368421052632|1\.4947368421052631|9\.551578947368421|2\.7305263157894735|0\.5621052631578948|2\.6736842105263157|0\.3031578947368421|6915\.44|3\.126315789473684|8\.303157894736842|5\.490526315789474|2\.6189473684210527|2\.7873684210526317|2\.741052631578947|0\.0|

Dari hasil rata-rata, dapat dilihat jika

- Klaster 1 rata-rata karyawannya berusia sekitar 47 tahun, klaster 2 berada di 38 tahun dan klaster 0 termuda di angka 32 tahun.
- Klaster 1 memiliki kepuasan lingkungan tertinggi dibanding lainnya, sementara kepuasan lingkungan terendah oleh klaster 0.
- Lembur paling banyak diambil oleh klaster 2. Namun, gaji tertinggi dimiliki oleh klaster 1, ini sejalan dengan lama tahun di perusahaan, klaster 1 paling lama yaitu rata-rata 6 tahun. Didukung memang hanya klaster 1 yang rata-ratanya karyawan senior.
- Sementara itu, rating performa karyawan dimana klaster 0 memiliki rating terbaik.
3. Visualisasi pengambilan lembur per klaster
  ![image](https://github.com/user-attachments/assets/055e75dc-b2f5-4171-a423-d226f0e34e41)
Terlihat jika banyak dari klaster 2 yang mengambil lembur, sementara yang paling banyak tidak mengambil lembur juga klaster 1
4. Visualisasi gaji bulanan per klaster
  ![image](https://github.com/user-attachments/assets/f1677313-e5ec-4164-9aa8-1699f752479a)
Terlihat dari boxplot, klaster 1 memiliki gaji bulanan paling banyak, sementara klaster 0 memiliki gaji yang paling sedikit, klaster 2 berada di atas klaster 0 sedikit lebih tinggi
5. Visualisasi kategori lain per klaster
  ![image](https://github.com/user-attachments/assets/583c606b-2004-4194-ae22-8de6d9784f39)
Dari hasil visualisasi di atas,
- klaster 0 didominasi oleh departemen 1,
- seluruh klaster jika dirata-rata, puas dengan lingkungan dan pekerjaan mereka,
- lembur palig banyak diambil klaster 0,
keterlibatan pekerjaan tertinggi (4) banyak dilakukan oleh klaster 0, dan
karyawan senior paling banyak di klaster 1.

## Business Dashboard

Jelaskan tentang business dashboard yang telah dibuat. Jika ada, sertakan juga link untuk mengakses dashboard tersebut.
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
