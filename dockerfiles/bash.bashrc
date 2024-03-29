export PS1="\[\e[31m\]upstride\[\e[m\]\[\033[01;32m\]@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ "
export TERM=xterm-256color

echo -e "\e[1;31m"
cat<<US
 _    _       _____ _        _     _
| |  | |     / ____| |      (_)   | |
| |  | |_ __| (___ | |_ _ __ _  __| | ___
| |  | | |_ \ ___ \| __|  __| |/ _\ |/ _ |
| |__| | |_) |___) | |_| |  | | (_| |  __/
 \____/| |__/_____/ \__|_|  |_|\__,_|\___|
       | |
       |_|

US
echo -e "\e[0;33m"

echo "
Welcome to UpStride Engine!
Any questions? hello@upstride.io

UpStride S.A.S, Paris, France -- upstride.io
"

if [[ $EUID -eq 0 ]]; then
  cat <<WARN
WARNING: You are running this container as root, which can cause new files in
mounted volumes to be created as the root user on your host machine.
To avoid this, run the container by specifying your user's userid:
$ docker run -u \$(id -u):\$(id -g) args...
WARN
else
  cat <<EXPL
You are running this container as user with ID $(id -u) and group $(id -g),
which should map to the ID and group for your user on the Docker host. Great!
EXPL
fi

# Turn off colors
echo -e "\e[m"

