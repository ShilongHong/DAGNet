import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class AutoDLClient:
    def __init__(self, token):
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        self.url = "https://api.autodl.com/api/v1/dev/deployment/container/stop"
        self.email_client = None

    def autodl_stop_container(self, deployment_container_uuid, decrease_one_replica_num=False, cmd_before_shutdown="sleep 5", best_status = None):


        body = {
            "deployment_container_uuid": deployment_container_uuid,
            "decrease_one_replica_num": decrease_one_replica_num,
            "cmd_before_shutdown": cmd_before_shutdown
        }
        
        response = requests.put(self.url, json=body, headers=self.headers)
        
        if response.status_code == 200:
            print("Container stopped successfully.")
            # 发送成功通知邮件
            self.send_email(f"The model best is {best_status}. and AutoDL Container Stop Success.", f"Container {deployment_container_uuid} stopped successfully.")
        else:
            print(f"Failed to stop container: {response.content.decode()}")
            # 发送失败通知邮件
            self.send_email("AutoDL Container Stop Failed", f"Failed to stop container {deployment_container_uuid}. Response: {response.content.decode()}")

    def initialize_email_client(self, smtp_server, smtp_port, username, password, from_addr, to_addr):
        """初始化邮件发送客户端"""
        self.email_client = {
            "smtp_server": smtp_server,
            "smtp_port": smtp_port,
            "username": username,
            "password": password,
            "from_addr": from_addr,
            "to_addr": to_addr
        }

    def send_email(self, subject, body):
        """发送邮件"""
        if not self.email_client:
            print("Error: 邮箱配置未初始化，无法发送邮件。")
            return

        msg = MIMEMultipart()
        msg["From"] = self.email_client["from_addr"]
        msg["To"] = self.email_client["to_addr"]
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        try:
            with smtplib.SMTP_SSL(self.email_client["smtp_server"], self.email_client["smtp_port"]) as server:
                server.login(self.email_client["username"], self.email_client["password"])
                server.sendmail(self.email_client["from_addr"], self.email_client["to_addr"], msg.as_string())
            print("Email sent successfully.")
        except Exception as e:
            print(f"Failed to send email: {str(e)}")


# 使用示例：
if __name__ == "__main__":
    token = "eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOjE3Njk3NywidXVpZCI6IjgwZjU3NjUxLWM0ZDQtNDAwYS1hMzNiLTFhOGZkMmVmNzUzZSIsImlzX2FkbWluIjpmYWxzZSwiYmFja3N0YWdlX3JvbGUiOiIiLCJpc19zdXBlcl9hZG1pbiI6ZmFsc2UsInN1Yl9uYW1lIjoiIiwidGVuYW50IjoiYXV0b2RsIiwidXBrIjoiIn0.x75OKtdqz3EmMbENYb-hOcamhZziLlQikTKm9bpWBaXI8GHx1Hlco5vWdhd-kbUkLI_vg6CttCDu_dJeHQcOvQ"
    client = AutoDLClient(token)
    
    # # 初始化邮件发送功能
    # client.initialize_email_client(
    #     smtp_server="smtp.example.com",
    #     smtp_port=465,
    #     username="your_email@example.com",
    #     password="your_email_password",
    #     from_addr="your_email@example.com",
    #     to_addr="recipient_email@example.com"
    # )
    
    # 假设这是您要停止的容器UUID
    deployment_container_uuid = "b4b74c8f6c-aaac1550"
    client.autodl_stop_container(deployment_container_uuid)
