echo "table_add MyIngress.ipv4_lpm MyIngress.forward 192.168.1.10/32 => 00:00:00:00:00:01 0" | simple_switch_CLI
echo "table_add MyIngress.ipv4_lpm MyIngress.forward 192.168.2.0/24 => 00:00:00:00:00:04 1" | simple_switch_CLI
echo "table_add MyIngress.ipv4_lpm MyIngress.forward 192.168.3.0/24 => 00:00:00:00:00:09 2" | simple_switch_CLI