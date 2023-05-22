$(document).ready(function(){

   // Fungsi untuk memeriksa apakah ada gambar yang diunggah
   function checkImageUploaded() {
    var fileSelected = $('#inputImage').prop('files').length > 0;
    $("#prediksi_submit").prop("disabled", !fileSelected);
  }

  // Memanggil fungsi checkImageUploaded saat ada perubahan pada unggahan gambar
  $('#inputImage').on("change", function() {
    checkImageUploaded();
  });

  // Memeriksa status unggahan gambar saat halaman dimuat
  checkImageUploaded();

  // Fungsi untuk memanggil API ketika tombol prediksi ditekan
  $("#prediksi_submit").click(function(e) {
    e.preventDefault();

    // Cek kembali apakah ada gambar yang diunggah
    var fileSelected = $('#inputImage').prop('files').length > 0;
    if (!fileSelected) {
      alert("Please upload images for analysis!");
      return;
    }

  });

    // Fungsi untuk memeriksa apakah ada skin tone yang dipilih
  function checkSkinToneSelected() {
    var selectedValue = $('#selectSkinTone').val();
    $("#prediksi_submit").prop("disabled", !selectedValue);
  }

  // Memanggil fungsi checkSkinToneSelected saat ada perubahan pada pemilihan skin tone
  $('#selectSkinTone').on("change", function() {
    checkSkinToneSelected();
  });

  // Memeriksa status pemilihan skin tone saat halaman dimuat
  checkSkinToneSelected();

  // Fungsi untuk memanggil API ketika tombol prediksi ditekan
  $("#prediksi_submit").click(function(e) { 
    e.preventDefault();

    // Cek kembali apakah ada skin tone yang dipilih
    var selectedValue = $('#selectSkinTone').val();
    if (!selectedValue) {
      alert("Please choose your skin tone first!");
      return;
    }

    // Fungsi untuk memanggil API ketika tombol prediksi ditekan
    $("#prediksi_submit").click(function(e) {
      e.preventDefault();
      
      // Get File Gambar yg telah diupload pengguna
      var file_data = $('#inputImage').prop('files')[0];   
      var select_data = $('#inputImage').prop('files')[1];   
      var pics_data = new FormData();                  
      pics_data.append('file', file_data);
      pics_data.append('selectSkinTone', select_data)

      var selectElement = document.getElementById("selectSkinTone");
      var selectedValue = selectElement.value;
  
      // Panggil API dengan timeout 1 detik (1000 ms)
      setTimeout(function() {
        try {
              $.ajax({
                  url : "/skinTone",
                  type : "POST",
                  data : { value: selectedValue },
                  success: function(response) {
                    // Handle the response from the server
                    console.log("Succes");
                  },
                  error: function(xhr, status, error) {
                    // Handle the error
                    console.log(error);
                  }
              });
          }
          catch(e) {
              // Jika gagal memanggil API, tampilkan error di console
              console.log("Gagal !");
              console.log(e);
          } 
      }, 1000)

      setTimeout(function() {
        try {
              $.ajax({
                  url         : "/api/faceDetect",
                  type        : "POST",
                  data        : pics_data,
                  processData : false,
                  contentType : false,
                  success     : function(res){
                      // Ambil hasil prediksi dan path gambar yang diprediksi dari API
                      res_data_prediksi   = res['prediksi']
                      res_gambar_prediksi = res['gambar_prediksi']
                      res_data_rekomendasi = res['data_rekomendasi']
                      
                      // Tampilkan hasil prediksi ke halaman web
                      generate_prediksi(res_data_prediksi, res_gambar_prediksi);
                      // Rekomennya masih belum
                      generate_recomenn(res_data_rekomendasi)
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

  });

    // -[Prediksi Model]---------------------------

    // Fungsi untuk menampilkan hasil prediksi analysis pada wajah
    function generate_prediksi(data_prediksi, image_prediksi) {
      var str="";
      
      if(image_prediksi == "(none)") {
        str += "<h3>Your Image is error</h3>";
        str += "<img src='https://dummyimage.com/250x250/000/fff' alt='Gambar Produk'>";
      }
      else {
        str += "<p>Your Problem on Face: <b>"+ data_prediksi +"</b></p>";
        str += "<img src='" + image_prediksi + "'width=\"300\" height=\"300\" alt='Gambar Produk'>";
      }
      $("#outputAreaFace").html(str);
    }

    // Fungsi untuk menampilkan hasil prediksi analysis pada wajah
    function generate_recomenn(data_rekomendasi) {
      var str="";
      if(data_rekomendasi[0]["Product_url"] == "(none)") {
        str += "<h3>Your Image is error</h3>";
        str += "<img src='https://dummyimage.com/250x250/000/fff' alt='Gambar Produk'>";
      }
      else {
        str += "<p><b>"+ data_rekomendasi[0]["Product"] +"</b></p>";
        str += "<img src='" + data_rekomendasi[0]["Product_url"] + "'width=\"250\" height=\"250\" alt='Gambar Produk'>";
      }
      $("#outputAreaProduk").html(str);
    }    
  })
  
