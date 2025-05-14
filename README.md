Bir Babanın İlhamı
Tırmanış sporu, hem fiziksel dayanıklılığı hem de zihinsel konsantrasyonu eş zamanlı geliştiren çok özel bir uğraş. Küçük kızım Sofia, bu spora büyük bir tutkuyla sarıldı. Ve Türkiye 1. si olarak bizleri onurlandırdı.  Onun gelişimini izlemek, her hamlesinde kararlılığını görmek benim için hem gurur verici hem de ilham verici oldu. Ancak bir mühendis ve teknoloji meraklısı bir baba olarak, sadece izlemek yetmedi. Ona dijital bir avantaj sunabilir miyim diye düşündüm. Ve böylece, “tırmanış analizi için görüntü işleme” temelli projem doğdu.

![Alt text](images/11.png)

Temel Hedefim Performansını Anlamak ve Desteklemekti.
Amacım basitti: Sofia'nın tırmanış esnasındaki hareketlerini analiz ederek, hangi bölgelerde zorlandığını, ne kadar sürede zirveye ulaştığını ve genel hareket desenlerini dijital olarak anlamaktı. Bu analizleri görsel olarak da destekleyerek hem onunla hem de antrenörleriyle verimli bir şekilde paylaşmak istedim.

Kullandığım sistem, Ultralytics’in YOLOv8-pose modeli üzerine kurulu. Bu model, bir kişinin vücut iskeletini tanıyıp anahtar noktaları (keypoints) tespit edebiliyor — örneğin, omuzlar, dirsekler, kalçalar, dizler ve ayak bilekleri.

çalışmamın kodunu github linkinden paylaşıyorum. kodun kısa özeti ise:

Video Analizi Başlatılıyor: input_video.mp4 dosyası açılıyor, çözünürlüğü ve FPS değeri alınıyor. Eğer dikey çekilmişse otomatik olarak döndürülüyor.

Poz Algılama: Her karede, tırmanıcıların kalça noktaları (left hip ve right hip) tespit ediliyor. Bu, kişinin genel konumunu belirlemek açısından ideal bir referans noktası.

Takip Mekanizması: Sistem, her karedeki kalça noktasını önceki kare ile karşılaştırarak takip edilen kişiyi tanımlıyor. Bu sayede Sofia’yı diğer figürlerden ayırmak mümkün oluyor.

Hareket Yolu ve İskelet Çizimi: Sofia’nın tırmanış sırasında oluşturduğu hareket hattı çiziliyor ve iskelet bağlantıları ile detaylı bir görsel analiz sunuluyor.

Çıktı Videosu: Tüm bu analizler, yeni bir video dosyası (output_pose_v5.mp4) olarak kaydediliyor. Böylece tüm tırmanış süreci hem ölçülebilir hem de izlenebilir hale geliyor.

![Alt text](images/3.png)


Zorlandığı Noktaların Tespiti 
Bu ilk versiyon daha çok poz takibi ve yol izleme üzerine kurulu. Ancak planladığım bir sonraki aşama, ısı haritası çıkararak Sofia’nın en çok zorlandığı bölgelere odaklanmak. Bu, frame başına kalça noktası hareket yoğunluğu, duraksama süresi gibi bilgilerle daha gelişmiş bir analiz yapma şansına sahip olabiliyorum.

Kod teknik görünebilir ama aslında yaptığı iş gayet insani:

Tırmanan kişinin “kalça noktası” ile merkezini buluyoruz. Çünkü bu nokta, vücut hareketini en iyi temsil ediyor.

Kalça noktalarını kare kare takip ederek bir yol oluşturuyoruz — tıpkı bir kuşun gökyüzündeki izini çizmek gibi.

Diğer anahtar noktalardan, vücut pozisyonunu ve duruşunu çıkarıyoruz — adeta dijital bir karakalem portresi gibi.

Bütün bu bilgileri hem video üzerinde çiziyor hem de saklıyoruz.

![Alt text](images/perf1.jpg)


Bu proje benim için sadece bir mühendislik tatmini değil, aynı zamanda bir ebeveyn olarak çocuğumun gelişimine olan katkım. Tırmanış gibi bireysel sporlarda, küçük geri bildirimler bile büyük fark yaratabilir. Dijital araçlar sayesinde bu geri bildirimleri veri ile desteklemek mümkün hale geliyor.

Bu proje benim için, yapay zeka, bilgisayarla görme ve babalık gibi üç ayrı alanın birleşimi oldu. Kızımın gelişimini izlemek sadece duygusal bir yolculuk değil, aynı zamanda teknolojik bir keşif halini aldı. Sofia tırmanmaya devam ettikçe ben de bu dijital yolda onunla birlikte tırmanıyor gibiyim.
