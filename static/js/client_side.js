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
    
    // Get File Gambar yg telah diupload pengguna
    var file_data = $('#inputImage').prop('files')[0];     
    var pics_data = new FormData();                  
    pics_data.append('file', file_data);

    // Panggil API dengan timeout 1 detik (1000 ms)

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
                    res_data_prediksi    = res['prediksi']
                    res_data_diagnosis   = res['diagnosis']
                    res_data_akurasi     = res['akurasi']
                    res_gambar_prediksi  = res['gambar_prediksi']
                    res_data_rekomendasi = res['data_rekomendasi']
                    
                    // Tampilkan hasil prediksi ke halaman web
                    generate_prediksi(res_data_prediksi, res_data_diagnosis, res_data_akurasi, res_gambar_prediksi);
                    generate_recomm(res_data_rekomendasi)
              }
            });
        }
        catch(e) {
            // Jika gagal memanggil API, tampilkan error di console
            console.log("Gagal !");
            console.log(e);
        } 
    }, 1000)
    
  });


  // -[Prediksi Model]---------------------------

  // Fungsi untuk menampilkan hasil prediksi analysis pada wajah
  function generate_prediksi(data_prediksi, data_diagnosis, data_akurasi, image_prediksi) {
    var str="";
    
    if(image_prediksi == "(none)") {
      str += "<h3>Your Image is error</h3>";
      str += "<img src='https://dummyimage.com/250x250/000/fff' alt='Gambar Produk'>";
    }
    else if(data_prediksi == "Upload JPG file"){
      str += "<p>Your page Error: <b>"+ data_prediksi +"</b></p>";
      str += "<img src='https://dummyimage.com/250x250/000/fff' alt='Gambar Produk'>";
    }
    else if(data_prediksi == "Normal / Not Detect"){
      str += "<p><b>"+ data_prediksi +"</b></p>";
      str += "<p>Hipotesis: <b>"+ data_diagnosis +"</b></p>";
      str += "<p>Accuration: <b>"+ data_akurasi +"</b></p>";
      str += "<img src='" + image_prediksi + "'width=\"300\" height=\"300\" alt='Gambar Produk'>";
    }
    else {
      str += "<p>Your Problem on Face: <b>"+ data_prediksi +"</b></p>";
      str += "<p>Accuration: <b>"+ data_akurasi +"</b></p>";
      str += "<img src='" + image_prediksi + "'width=\"300\" height=\"300\" alt='Gambar Produk'>";
    }
    $("#outputAreaFace").html(str);
  }

  function generate_recomm(data_recom) {
    var str="";
    // Membuat list untuk data medications
    var medications = data_recom[0]["Medication"].map(function(medication) {
      return '<li>' + medication + '</li>';
    });
    var listMed = medications.join('')

    // Membuat list untuk data ingredients
    var ingredients = data_recom[0]["Skincare Ingredients"].map(function(ingredient) {
      return '<li>' + ingredient + '</li>';
    });
    var listIngr = ingredients.join('')

    // Membuat list untuk data resources
    var resources = data_recom[0]["Domain"].map(function(resource, index) {
      return '<li><a href="'+ data_recom[0]["Resources"][index] +'" target="_blank">' + resource + '</a></li>';
    });
    var listRes = resources.join('')
    
    if(data_recom == "(none)") {
      str +=  '<td class="text-start" style="border-right: 2px solid #3D405B;">'
      str +=  '<ul style="text-align: justify;">'
      str +=  '<li>Ketoconazole cream Lorem ipsum dolor sit amet consectetur adipisicing elit.</li>'
      str +=  '<li>Clotrimazole cream</li>'
      str +=  '<li>Miconazole cream</li>'
      str +=  '<li>Selenium sulfide shampoo</li>'
      str +=  '<li>Nystatin powder</li>'
      str +=  '</ul>'
      str +=  '</td>'
      str +=  '<td class="no-bullet text-start">'
      str +=  '<ul style="text-align: justify;">'
      str +=  '<li>Salicylic acid</li>'
      str +=  '<li>Tea tree oil</li>'
      str +=  '<li>Niacinamide</li>'
      str +=  '<li>Zinc pyrithione</li>'
      str +=  '<li>Aloe vera</li>'
      str +=  '</ul>'
      str +=  '</td>'
      
    }
    else {
      str +=  '<td class="text-start" style="border-right: 2px solid #3D405B;">'
      str +=  '<ul style="text-align: justify;">'
      str +=  listMed
      str +=  '</ul>'
      str +=  '</td>'
      str +=  '<td class="text-start" style="border-right: 2px solid #3D405B;">'
      str +=  '<ul style="text-align: justify;">'
      str +=  listIngr
      str +=  '</ul>'
      str +=  '</td>'
      str +=  '<td class="text-start">'
      str +=  '<ul style="text-align: justify;">'
      str +=  listRes
      str +=  '</ul>'
      str +=  '</td>'
    }


    $("#outputRecomm").html(str);
  }

  })
  
