while : ; do
    python3 depthai_demo.py "$@"
    exit_code=$?
    echo "Exit code: $exit_code"
    if [ $exit_code -le 4 ]; then
      break
    fi
done

exit $exit_code
