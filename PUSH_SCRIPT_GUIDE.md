# push_to_private.sh - 使用指南

这是一个增强版的推送脚本，支持SSH和HTTPS两种认证方式，自动检测并推荐最佳方式。

## 功能特性

✅ **智能检测**: 自动检测SSH密钥配置
✅ **双模式支持**: SSH和HTTPS两种认证方式
✅ **彩色输出**: 清晰的提示和状态信息
✅ **错误处理**: 详细的错误提示和解决方案
✅ **安全验证**: 推送前确认配置

---

## 使用方法

### 基本用法

```bash
./push_to_private.sh USERNAME [REPO_NAME]
```

### 参数说明

- `USERNAME`: 必需 - 你的GitHub用户名
- `REPO_NAME`: 可选 - 仓库名称（默认: ml-clara-private）

### 使用示例

```bash
# 示例 1: 使用默认仓库名
./push_to_private.sh xucheng

# 示例 2: 指定仓库名
./push_to_private.sh xucheng ml-clara-rag

# 示例 3: 不同的仓库名
./push_to_private.sh johndoe my-awesome-project
```

---

## 认证方式选择

脚本启动后会显示两种认证方式供选择：

### 选项 1: SSH（推荐）

**优点：**
- ✅ 无需每次输入密码
- ✅ 更安全
- ✅ 配置一次，永久使用

**适用场景：**
- 已配置SSH密钥
- 经常需要推送代码
- 追求便捷性

**示例输出：**
```
Choose authentication method:

  1) SSH (Recommended - No password needed)
     URL: git@github.com:xucheng/ml-clara-rag.git

  2) HTTPS (Requires Personal Access Token)
     URL: https://github.com/xucheng/ml-clara-rag.git

✓ SSH key detected - Option 1 recommended

Enter choice (1 or 2) [default: 1]:
```

### 选项 2: HTTPS

**优点：**
- ✅ 无需SSH配置
- ✅ 适合临时使用
- ✅ 防火墙兼容性好

**适用场景：**
- 未配置SSH密钥
- 临时设备
- 网络环境限制SSH

**示例输出：**
```
Choose authentication method:

  1) SSH (Requires SSH key setup)
     URL: git@github.com:xucheng/ml-clara-rag.git

  2) HTTPS (Requires Personal Access Token)
     URL: https://github.com/xucheng/ml-clara-rag.git

⚠️  No SSH key detected - Option 2 recommended

Enter choice (1 or 2) [default: 2]:
```

---

## 完整流程演示

### 使用SSH方式

```bash
$ ./push_to_private.sh xucheng ml-clara-rag

=========================================
  Pushing to Private Repository
=========================================

GitHub Username: xucheng
Repository Name: ml-clara-rag

Choose authentication method:

  1) SSH (Recommended - No password needed)
     URL: git@github.com:xucheng/ml-clara-rag.git

  2) HTTPS (Requires Personal Access Token)
     URL: https://github.com:xucheng/ml-clara-rag.git

✓ SSH key detected - Option 1 recommended

Enter choice (1 or 2) [default: 1]: 1

Testing SSH connection to GitHub...
✓ SSH authentication successful

=========================================
Configuration Summary
=========================================
Username:     xucheng
Repository:   ml-clara-rag
Auth Method:  SSH
Remote URL:   git@github.com:xucheng/ml-clara-rag.git
=========================================

Proceed with these settings? (y/n): y

Removing old origin...
✓ Old origin removed

Adding new origin...
✓ New origin added

origin  git@github.com:xucheng/ml-clara-rag.git (fetch)
origin  git@github.com:xucheng/ml-clara-rag.git (push)

Pushing to private repository...
Enumerating objects: 117, done.
Counting objects: 100% (117/117), done.
...
To github.com:xucheng/ml-clara-rag.git
 * [new branch]      main -> main

=========================================
  ✅ Successfully Pushed!
=========================================

Repository URL: https://github.com/xucheng/ml-clara-rag

Next steps:
  1. Verify at: https://github.com/xucheng/ml-clara-rag
  2. Check that it shows 🔒 Private status
  3. Review README.md is displayed correctly

Clone this repository:
  git clone git@github.com:xucheng/ml-clara-rag.git
```

### 使用HTTPS方式

```bash
$ ./push_to_private.sh xucheng ml-clara-rag

[选择选项 2]

=========================================
⚠️  Authentication Required
=========================================

You will be prompted for credentials:
  Username: xucheng
  Password: YOUR_PERSONAL_ACCESS_TOKEN

⚠️  Use Personal Access Token, NOT your GitHub password!

Get your token at:
  https://github.com/settings/tokens

Token setup:
  1. Click 'Generate new token (classic)'
  2. Select 'repo' scope
  3. Copy the generated token

Press Enter when ready to push...

[按Enter后会提示输入用户名和token]
```

---

## SSH密钥设置（首次使用）

如果选择SSH方式但未配置密钥，请按以下步骤操作：

### 1. 生成SSH密钥

```bash
# 生成新的SSH密钥
ssh-keygen -t ed25519 -C "your_email@example.com"

# 按Enter使用默认路径
# 可选：设置密码短语（推荐）
```

### 2. 复制公钥

```bash
# macOS
cat ~/.ssh/id_ed25519.pub | pbcopy

# Linux
cat ~/.ssh/id_ed25519.pub
# 然后手动复制输出内容
```

### 3. 添加到GitHub

1. 访问: https://github.com/settings/keys
2. 点击 "New SSH key"
3. Title: 填写描述（如 "MacBook Pro"）
4. Key: 粘贴公钥内容
5. 点击 "Add SSH key"

### 4. 测试连接

```bash
ssh -T git@github.com

# 成功输出示例:
# Hi xucheng! You've successfully authenticated, but GitHub does not provide shell access.
```

---

## Personal Access Token设置

如果选择HTTPS方式，需要Personal Access Token：

### 1. 创建Token

1. 访问: https://github.com/settings/tokens
2. 点击 "Generate new token" → "Tokens (classic)"
3. 填写信息:
   - **Note**: `ml-clara-deployment`
   - **Expiration**: 90 days（或No expiration）
   - **Scopes**: ✅ `repo` (完整仓库控制)
4. 点击 "Generate token"
5. **⚠️ 立即复制token并保存**（只显示一次）

### 2. 使用Token

推送时输入：
- Username: `xucheng`
- Password: `ghp_xxxxxxxxxxxxxxxxxxxx`（你的token）

### 3. 保存Token（可选）

使用Git凭据管理器避免重复输入：

```bash
# macOS
git config --global credential.helper osxkeychain

# Linux
git config --global credential.helper cache

# Windows
git config --global credential.helper wincred
```

---

## 常见问题

### Q1: 脚本提示"No SSH key detected"但我已配置？

**解决方案:**
```bash
# 测试SSH连接
ssh -T git@github.com

# 如果失败，检查SSH agent
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

### Q2: SSH推送失败，提示"Permission denied"？

**可能原因:**
- SSH密钥未添加到GitHub
- SSH密钥文件权限错误

**解决方案:**
```bash
# 检查密钥权限
chmod 600 ~/.ssh/id_ed25519
chmod 644 ~/.ssh/id_ed25519.pub

# 重新测试
ssh -T git@github.com
```

### Q3: HTTPS推送失败，提示"Authentication failed"？

**可能原因:**
- Token错误或过期
- Token权限不足

**解决方案:**
1. 重新生成token
2. 确保勾选了 `repo` 权限
3. 检查token是否过期

### Q4: 如何切换认证方式？

**解决方案:**
```bash
# 查看当前remote
git remote -v

# 切换到SSH
git remote set-url origin git@github.com:USERNAME/REPO.git

# 切换到HTTPS
git remote set-url origin https://github.com/USERNAME/REPO.git
```

### Q5: 推送很慢怎么办？

**解决方案:**

SSH方式：
```bash
# 编辑 ~/.ssh/config
Host github.com
    Hostname ssh.github.com
    Port 443
```

HTTPS方式：
```bash
# 使用代理（如有）
git config --global http.proxy http://127.0.0.1:7890
git config --global https.proxy https://127.0.0.1:7890
```

---

## 脚本特性详解

### 1. 智能检测

脚本自动检测SSH配置：
- 测试GitHub SSH连接
- 检查常见SSH密钥文件
- 根据结果推荐认证方式

### 2. 彩色输出

- 🟢 绿色: 成功消息
- 🟡 黄色: 警告和提示
- 🔴 红色: 错误信息
- 🔵 蓝色: 信息标题

### 3. 安全验证

推送前显示配置摘要，确保：
- 用户名正确
- 仓库名正确
- 认证方式合适
- Remote URL准确

### 4. 详细错误提示

失败时提供：
- 可能的原因
- 具体的解决方案
- 相关链接

---

## 高级用法

### 自动化脚本集成

```bash
#!/bin/bash
# 在CI/CD中使用

# 设置变量
export GITHUB_USERNAME="xucheng"
export REPO_NAME="ml-clara-rag"

# 非交互模式（需要预先配置SSH）
git remote remove origin 2>/dev/null || true
git remote add origin git@github.com:$GITHUB_USERNAME/$REPO_NAME.git
git push -u origin main
```

### 批量推送多个仓库

```bash
#!/bin/bash

repos=("repo1" "repo2" "repo3")

for repo in "${repos[@]}"; do
    echo "Pushing to $repo..."
    ./push_to_private.sh xucheng "$repo"
done
```

---

## 相关文档

- **DEPLOY_GUIDE.md** - 完整部署指南
- **README.md** - 项目使用文档
- **.env.example** - 环境变量配置

---

## 技术支持

遇到问题？
1. 查看本文档的"常见问题"部分
2. 阅读 DEPLOY_GUIDE.md
3. 检查GitHub文档: https://docs.github.com

---

**版本**: 2.0 (SSH + HTTPS支持)
**最后更新**: 2025-12-01
