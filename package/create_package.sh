SOURCE_DIR=$(dirname "$0")
PACKAGE_NAME="$1"
cd "$SOURCE_DIR"

# Add all files except dependencies
tar --directory='../' --xform 's,^./,depthai/,' --exclude='./package' --exclude='.git' --exclude='./depthai-api/shared/3rdparty' -cf "$PACKAGE_NAME.tar" .

# Append dependencies but only needed parts
tar --directory='../' --xform 's,^./,depthai/,' -rf "$PACKAGE_NAME.tar" ./depthai-api/shared/3rdparty/dldt/inference-engine/thirdparty/movidius/XLink ./depthai-api/shared/3rdparty/json-schema-validator ./depthai-api/shared/3rdparty/boost_1_71_0/boost

# Append nlohmann json without tests
tar --directory='../' --xform 's,^./,depthai/,' --exclude='./depthai-api/shared/3rdparty/json/test' -rf "$PACKAGE_NAME.tar" ./depthai-api/shared/3rdparty/json

# Compress into tar.gz
gzip -c "$PACKAGE_NAME.tar" > "$PACKAGE_NAME.tar.gz"
