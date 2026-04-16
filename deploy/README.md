# Deployment

## Mac Studio (launchd)

```bash
cp deploy/launchd/cc.saillant.micro-kiki.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/cc.saillant.micro-kiki.plist
```

## kxkm-ai (systemd)

```bash
sudo cp deploy/systemd/micro-kiki.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now micro-kiki
```

## Logs

- Mac: `tail -f /tmp/micro-kiki.stderr.log`
- Linux: `journalctl -u micro-kiki -f`
