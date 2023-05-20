$(document).ready(function(){

    // -[Prediksi Model]---------------------------
    
    // Fungsi untuk memanggil API ketika tombol prediksi ditekan
    $("#prediksi_submit").click(function(e) {
      e.preventDefault();
      
      // Get File Gambar yg telah diupload pengguna
      var file_data = $('#inputImage').prop('files')[0];   
      var pics_data = new FormData();                  
      pics_data.append('file', file_data);
  
      // Panggil API dengan timeout 1 detik (1000 ms)
      setTimeout(function() {
        try {
              $.ajax({
                  url         : "/api/deteksi",
                  type        : "POST",
                  data        : pics_data,
                  processData : false,
                  contentType : false,
                  success     : function(res){
                      // Ambil hasil prediksi dan path gambar yang diprediksi dari API
                      res_data_prediksi   = res['prediksi']
                      res_gambar_prediksi = res['gambar_prediksi']
                      
                      // Tampilkan hasil prediksi ke halaman web
                      generate_prediksi(res_data_prediksi, res_gambar_prediksi);
                      // Rekomennya masih belum
                      generate_recomenn(res_data_prediksi, res_gambar_prediksi)
                }
              });
          }
          catch(e) {
              // Jika gagal memanggil API, tampilkan error di console
              console.log("Gagal !");
              console.log(e);
          } 
      }, 1000)  
    })
     
    // Fungsi untuk menampilkan hasil prediksi model
    function generate_prediksi(data_prediksi, image_prediksi) {
      var str="";
      
      if(image_prediksi == "(none)") {
        str += "<h3>Your Image is error</h3>";
        str += "<img src='https://dummyimage.com/300x300/000/fff' alt='Gambar Produk'>";
      }
      else {
        str += "<p>Your Problem on Face: <b>"+ data_prediksi +"</b></p>" 
        str += "<img src='" + image_prediksi + "'width=\"350\" height=\"350\" alt='Gambar Produk'>";
      }
      $("#outputAreaFace").html(str);
    }

    // Fungsi untuk menampilkan hasil prediksi rekomendasi
    function generate_recomenn(product_prediction, product_image) {
      var str="";
      
      if(product_image == "(none)") {
        str += "<h3>Your Image is error</h3>";
        str += "<img src='https://dummyimage.com/300x300/000/fff' alt='Gambar Produk'>";
      }
      else {
        str += "<p>Product on Your Face: <b>"+ product_prediction +"</b></p>" 
        str += "<img src='" + product_image + "'width=\"350\" height=\"350\" alt='Gambar Produk'>";
      }
      $("#outputAreaFace").html(str);
    }  
  })
    
  