<!DOCTYPE html>
<html>
<head>
    <title>异常视频检测平台 V1.0</title>
</head>

<body>

<div class="login-page">
    <div clss="header">
        <div class="logo">
            <img src="../static/url/logo_new2.png" class>

        </div>
        <div class="login-container">
            <div class="login-box">
                <h1>异常视频检测平台</h1>
            {% if error %}
                <p>{{ error }}</p>
            {% endif %}
            <form method="POST" action="/login">
                <label for="username">用户:</label>
                <input type="text" id="username" name="username" required><br>
                <label for="password">密码:</label>
                <input type="password" id="password" name="password" required>
                <br>
                <br>
                <div class="go">
                <input type="submit" value="登陆">
                </div>
            </form>

            </div>
        </div>
        <div class="put">

        </div>
    </div>
</div>
</body>
</html>