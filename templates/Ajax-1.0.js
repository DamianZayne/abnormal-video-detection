function  Ajax(obj){
    //定义默认参数
    var defaults={
        url:"#",
        type:"get",
        datetype:"text",
        data:{},
        success:function(d){
            console.log(d);
        },
        async:true,
    };

    
    //用户传输过来的数据，更改defaults默认值
    for(var k in obj){
        defaults[k]=obj[k];
    }

    //判断浏览器的版本，再来创建网咯对象
    var xhr=null;
    if(window.XMLHttpRequest){
        xhr=new XMLHttpRequest();
    }
    else{
        xhr=new ActiveXObject("Microsoft.XMLHTTP");
    }

    //判断用户是否传了数据，并且做拼接
    var dataurl="";
    if(defaults.data){
        var value=defaults.data
        dataurl+="?";
        for(var k in defaults.data){
            
            dataurl+=k+"="+value[k]+"&";
        }
        dataurl.slice(0,dataurl.length-1);
    }

    
    
    //判断type是get还是post
    if(defaults.type=="get"){
        defaults.url+="?"+dataurl;
    }

    //写好配置并发送
    xhr.open(defaults.type,defaults.url,defaults.async);

    //post请求发送，参数用xhr.send()发送
    var sendDate=null;
    if(defaults.type=="post"){
        sendDate=dataurl;
        xhr.setRequestHeader("Content-Type","application/x-www-from-urlencoded");
        // Coutent-Type,application/x-www-from-urlencoded
    }
    xhr.send(sendDate);



    //等待，接收返回信息
    xhr.onreadystatechange=function()
    {
        if(xhr.readyState==4&&xhr.status==200)
        {

            //判断defaults.dataType是什么格式
            var data=xhr.responseText;
            if(defaults.datetype=="json"){
                data=JSON.parse(data);
            }
            defaults.success(data);
        }
    }

        
}