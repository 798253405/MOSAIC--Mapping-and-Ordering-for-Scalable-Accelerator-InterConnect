#!/bin/bash

echo "======================================"
echo "验证 Binary Switch 和 Fire Advance 修复"
echo "======================================"
echo ""

# 验证 Binary Switch 拼写修复
echo "1. 检查 Binary Switch 宏名拼写..."
echo "   应该全部是 'binaryroutingSwitch'（正确拼写）"
echo ""

FILES="src/MAC.hpp src/MAC.cpp src/llmmac.hpp src/llmmac.cpp"
for file in $FILES; do
    echo "   $file:"
    grep -n "ifdef.*routing\|ifndef.*routing" $file 2>/dev/null | head -3
done

echo ""
echo "2. 检查 parameters.hpp 中的定义..."
grep -n "define.*routing.*Switch" src/parameters.hpp | head -5

echo ""
echo "3. 检查 Fire Advance 实现 (CNN)..."
echo "   MAC.hpp 中的 Fire Advance 成员变量:"
grep -A3 "#ifdef fireAdvance" src/MAC.hpp | head -6

echo ""
echo "   MAC.cpp 中的 Fire Advance 逻辑:"
grep -c "ifdef fireAdvance" src/MAC.cpp
echo "   找到 $(grep -c 'ifdef fireAdvance' src/MAC.cpp) 处 Fire Advance 代码块"

echo ""
echo "4. 编译测试..."
cd Debug
make clean > /dev/null 2>&1
echo "   正在编译..."
if make all > /tmp/compile.log 2>&1; then
    echo "   ✅ 编译成功！"
    ls -lh 2508date | awk '{print "   二进制大小:", $5}'
else
    echo "   ❌ 编译失败，错误信息："
    tail -20 /tmp/compile.log
    exit 1
fi

echo ""
echo "======================================"
echo "修复验证完成！"
echo "======================================"
