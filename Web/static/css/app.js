const sign_in_btn = document.querySelector("#sign-in-btn");
const sign_up_btn = document.querySelector("#sign-up-btn");
const container = document.querySelector(".container");
const input = document.querySelector("#btnloginiii");
const element   = document.querySelector(".container1"); 

const user  = document.querySelector(".Userkkk"); 
const user_info  = document.querySelector(".user-info"); 
const square_contain  = document.querySelector(".square-container-wrapper");
const order  = document.querySelector(".orders");
const statistics  = document.querySelector(".statistics");
sign_up_btn.addEventListener('click', () =>{
    container.classList.add("sign-up-mode");
    element.classList.remove("hidden");
});

sign_in_btn.addEventListener('click', () =>{
    container.classList.remove("sign-up-mode");
    element.classList.add("hidden");
});
//-----------------------------------------------------------------------------
// function sendUser(email) {
//     $.ajax({
//         type: 'POST',
//         url: '/send_user',
//         data: JSON.stringify({ email: email }),
//         contentType: 'application/json',
//         success: function(response) {
//             var img=$('<img>').addClass('avatartl');
//             img.attr('src', response.avater_url);
//             var div=$('<div>').addClass('user-details');
//             var name=$('<h3>').text(response.name);    
//             var job=$('<p>').text(response.job);
//             var sex=$('<p>').text(response.sex);
//             var addr=$('<p>').text(response.addr);
//             var dob=$('<p>').text(response.dob);
//             if(response.name==""){
//                 name.text("Chưa có");
//             }
//             if(response.dob==""){
//                 dob.text("Chưa có");
//             }
//             if(response.job==""){
//                 job.text("Chưa có");
//             }
//             if(response.addr==""){
//                 addr.text("Chưa có");
//             }
//             if(response.sex==""){
//                 sex.text("Chưa có");
//             }
//             div.append(name);
//             div.append(sex);
//             div.append(dob);
//             div.append(addr);
//             div.append(job);
//             user_info.append(img);
//             user_info.append(div); 
//         },
//         error: function(xhr, status, error) {
//             var img=$('<img>').addClass('avatartl');//chưa set src cho ảnh
//             var div=$('<div>').addClass('user-details');
//             var name=$('<h3>').text("Không thể load thông tin người này");
//             div.append(name);
//             user_info.append(img);
//             user_info.append(div); 
//         }
//     });
// }