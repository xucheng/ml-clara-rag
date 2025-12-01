# 部署指南：将项目上传为私有GitHub仓库

## 方案一：创建新的私有仓库（推荐）

### 步骤1：提交本地更改

```bash
# 进入项目目录
cd /Users/xucheng/repo/ml-clara

# 添加所有更改
git add .

# 创建提交
git commit -m "feat: Add environment variable support and comprehensive documentation

- Replace hardcoded paths with environment variables
- Add .env.example for easy configuration
- Update README with data pipeline details and troubleshooting
- Add .gitignore to exclude temporary files
- Update training scripts to support flexible paths"
```

### 步骤2：在GitHub创建私有仓库

1. 访问 https://github.com/new
2. 填写仓库信息：
   - **Repository name**: `ml-clara-private` （或你喜欢的名字）
   - **Description**: CLaRa: Bridging Retrieval and Generation with Continuous Latent Reasoning
   - **选择**: ✅ Private（私有）
   - **不要**勾选：Initialize with README, .gitignore, license（我们已经有了）
3. 点击 "Create repository"

### 步骤3：添加新的远程仓库并推送

```bash
# 如果要完全替换现有的origin
git remote remove origin

# 添加新的私有仓库（替换为你的用户名和仓库名）
git remote add origin https://github.com/YOUR_USERNAME/ml-clara-private.git

# 推送代码
git branch -M main
git push -u origin main
```

---

## 方案二：保留原仓库，添加私有远程仓库

如果你想保留原仓库的连接，同时推送到私有仓库：

```bash
# 重命名原来的origin为upstream
git remote rename origin upstream

# 添加新的私有仓库
git remote add origin https://github.com/YOUR_USERNAME/ml-clara-private.git

# 推送到私有仓库
git branch -M main
git push -u origin main

# 以后可以从原仓库拉取更新
git pull upstream main
```

---

## 方案三：导出为压缩包（无git历史）

如果你想要一个干净的副本（不包含git历史）：

```bash
# 创建压缩包（排除git历史和临时文件）
cd /Users/xucheng/repo
tar -czvf ml-clara-clean.tar.gz \
    --exclude='.git' \
    --exclude='*.pyc' \
    --exclude='__pycache__' \
    --exclude='example/extracted_assets' \
    --exclude='checkpoints' \
    --exclude='wandb' \
    --exclude='.DS_Store' \
    ml-clara/

# 解压到新位置
mkdir -p ml-clara-new
tar -xzvf ml-clara-clean.tar.gz -C ml-clara-new

# 进入新目录并初始化git
cd ml-clara-new/ml-clara
git init
git add .
git commit -m "Initial commit: CLaRa project with environment variable support"

# 添加远程仓库并推送
git remote add origin https://github.com/YOUR_USERNAME/ml-clara-private.git
git branch -M main
git push -u origin main
```

---

## 验证推送

推送完成后，访问你的GitHub仓库页面验证：

```
https://github.com/YOUR_USERNAME/ml-clara-private
```

确认：
- ✅ 仓库显示为 🔒 Private
- ✅ README.md 正确显示
- ✅ .gitignore 正在工作（检查extracted_assets等文件夹没有被上传）
- ✅ 所有脚本和文档都已上传

---

## 注意事项

### 🔒 安全检查

在推送前，确保以下敏感信息**没有**包含在代码中：

```bash
# 检查是否有API密钥泄露
grep -r "sk-" . --exclude-dir=".git" --exclude="*.md"
grep -r "OPENAI_API_KEY=" . --exclude-dir=".git" --exclude=".env.example"

# 检查是否有个人路径
grep -r "/Users/xucheng" . --exclude-dir=".git" --exclude="*.md" --exclude="DEPLOY_GUIDE.md"
```

### 📦 推荐的.gitignore已包含

已添加的.gitignore会自动排除：
- ✅ Python缓存 (__pycache__, *.pyc)
- ✅ 虚拟环境 (venv/, env/)
- ✅ 训练输出 (checkpoints/, wandb/)
- ✅ 数据文件 (extracted_assets/)
- ✅ 环境变量 (.env)
- ✅ 系统文件 (.DS_Store)

### 🔄 保持更新

后续更新代码：

```bash
# 修改代码后
git add .
git commit -m "描述你的更改"
git push origin main

# 从原始仓库同步更新（如果使用方案二）
git pull upstream main
```

---

## 团队协作

如果需要与团队成员共享私有仓库：

1. 访问仓库设置：`https://github.com/YOUR_USERNAME/ml-clara-private/settings/access`
2. 点击 "Invite a collaborator"
3. 输入团队成员的GitHub用户名
4. 选择权限级别：
   - **Read**: 只能查看
   - **Write**: 可以提交代码
   - **Admin**: 完全控制

---

## 克隆私有仓库

团队成员克隆私有仓库：

```bash
# 使用HTTPS（需要输入GitHub用户名和密码/token）
git clone https://github.com/YOUR_USERNAME/ml-clara-private.git

# 或使用SSH（需要先配置SSH密钥）
git clone git@github.com:YOUR_USERNAME/ml-clara-private.git
```

---

## 常见问题

**Q: 推送时提示认证失败？**

从2021年起，GitHub不再支持密码认证。你需要：

1. 创建Personal Access Token (PAT):
   - 访问 https://github.com/settings/tokens
   - "Generate new token" → "Classic"
   - 勾选 `repo` 权限
   - 复制生成的token

2. 使用token作为密码：
   ```bash
   git push -u origin main
   # Username: YOUR_USERNAME
   # Password: ghp_xxxxxxxxxxxx（你的token）
   ```

**Q: 如何配置SSH避免每次输入密码？**

```bash
# 生成SSH密钥
ssh-keygen -t ed25519 -C "your_email@example.com"

# 添加到GitHub
# 1. 复制公钥
cat ~/.ssh/id_ed25519.pub
# 2. 访问 https://github.com/settings/keys
# 3. 点击 "New SSH key" 并粘贴

# 修改远程仓库为SSH
git remote set-url origin git@github.com:YOUR_USERNAME/ml-clara-private.git
```

**Q: 不小心提交了敏感信息怎么办？**

```bash
# 从历史记录中删除文件
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch path/to/sensitive/file" \
  --prune-empty --tag-name-filter cat -- --all

# 强制推送（谨慎使用！）
git push origin --force --all
```

**建议**: 如果泄露了API密钥，立即在服务商处撤销密钥！
