# Deployment

## Mac Studio (MLX serving)

```bash
# Stage the launchd plist (do NOT load in production without review)
cp deploy/launchd/cc.saillant.micro-kiki.plist ~/Library/LaunchAgents/
# To load: launchctl load ~/Library/LaunchAgents/cc.saillant.micro-kiki.plist
# To unload: launchctl unload ~/Library/LaunchAgents/cc.saillant.micro-kiki.plist
```

## Linux / kxkm-ai (vLLM serving)

```bash
# Stage the systemd unit (do NOT enable without review)
sudo cp deploy/systemd/micro-kiki.service /etc/systemd/system/
# To start: sudo systemctl start micro-kiki
# To enable on boot: sudo systemctl enable micro-kiki
```

## Aeon compression daemon

```bash
# Mac Studio
cp deploy/systemd/aeon-compress.service /etc/systemd/system/
# Or run manually:
uv run python scripts/aeon_compress_daemon.py --interval 3600 --max-age-days 30
```

## Docker (vLLM)

```bash
docker build -f docker/vllm.Dockerfile -t micro-kiki-vllm .
docker run --gpus all -p 8100:8100 micro-kiki-vllm
```
