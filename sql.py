from flask import Flask, render_template, redirect, url_for, request
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# 配置 Flask-Login
login_manager = LoginManager(app)
login_manager.login_view = 'login'  # 未登录时跳转到登录页

# 模拟用户数据库
class User(UserMixin):
    def __init__(self, id):
        self.id = id

users = {'admin': {'password': '123456'}}

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

# 登录页
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username in users and users[username]['password'] == password:
            user = User(username)
            login_user(user )  # 登录用户
            next_page = request.args.get('next')
            return redirect(next_page or url_for('index'))
        return "登录失败！"
    return '''
        <form method="post">
            用户名: <input type="text" name="username"><br>
            密码: <input type="password" name="password"><br>
            <button type="submit">登录</button>
        </form>
    '''

# 首页（公开访问）
@app.route('/')
def index():
    return '欢迎来到首页！<a href="/protected">进入受保护页面</a>'

# 受保护页面（必须登录）
@app.route('/up')
@login_required  # Flask-Login 提供的装饰器
def protected():
    return f'你好, {current_user.id}! 这是受保护页面。'

# 登出
@app.route('/logout')
@login_required
def logout():
    logout_user()  # 登出用户
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
'''
import redis

# 获取redis数据库连接
r = redis.StrictRedis(host="127.0.0.1", port=6379, db=0,decode_responses=True)
r.set (name="add1",value="的防晒服")
print(r.get("add1"))
'''
'''
# redis存入键值对
r.set(name="key", value="value")
# 读取键值对
print(r.get("key"))
# 删除
print(r.delete("key"))

# redis存入Hash值
r.hset(name="name", key="key1", value="value1")
r.hset(name="name", key="key2", value="value2")
# 获取所有哈希表中的字段
print(r.hgetall("name"))
# 获取所有给定字段的值
print(r.hmget("name", "key1", "key2"))
# 获取存储在哈希表中指定字段的值。
print(r.hmget("name", "key1"))
# 删除一个或多个哈希表字段
print(r.hdel("name", "key1"))

# 过期时间
r.expire("name", 60)  # 60秒后过期


'''
