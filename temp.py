

"""
Proje Amacı 
Bu projenin amacı, bir kameradan gelen gerçek zamanlı görüntüler üzerinde insanların 
sosyal mesafeye uyup uymadığını ve maske takıp takmadığını tespit eden bir sistem 
geliştirmektir. Sistem, YOLOv8 nesne algılama modeli ve OpenCV yardımıyla çalışır.

Projede Kullanılan Temel Kavramlar 
 Nesne Tespiti (Object Detection): İnsan ve maske tespiti için. 
 İzleme (Tracking): İstenirse insanları takip etmek için DeepSORT veya ByteTrack. 
 Mesafe Ölçümü: İki insanın görüntü üzerindeki konumlarına göre aralarındaki 
mesafenin yaklaşık tahmini. 
 Maske Tespiti: Eğitimli modelle veya YOLOv8 multi-class ile "maskeli" ve "maskesiz" 
ayrımı yapılır. 

Sistem Mimarisi ve İşleyişi 
1. Kamera görüntüsü alınır. 
2. YOLOv8 modeli ile her karede insan ve maske tespiti yapılır. 
3. İnsanlar arasındaki mesafeler hesaplanır (örneğin, Pythagorean teoremi ile). 
4. Mesafe belirli bir eşikten azsa, uyarı çerçevesi kırmızı olur. 
5. Maske takmayan kişiler ayrıca işaretlenir. 
6. Ekrana anlık uyarı mesajları veya renkli kutularla çıktı verilir.

Öğrenim Hedefleri 
Bu proje sayesinde öğrenci: 
 Derin öğrenme temelli nesne tespiti modellerini kullanmayı, 
 OpenCV ile gerçek zamanlı kare işleme yapmayı,  
 Görüntü üzerinde konum verisini analiz etmeyi, 
 İleri düzey proje mantığı ve performans optimizasyonunu öğrenir. 
Teslim Edilecekler 
 Kod dosyaları 
 Kendi yorumlarını içeren açıklamalı bir PDF 
 Proje çıktılarının ekran görüntüleri 
 (Opsiyonel) Modeli eğitmek için kullanılan veri seti  
"""



