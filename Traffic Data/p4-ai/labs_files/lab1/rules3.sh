echo "table_add MyIngress.ipv4_lpm MyIngress.forward 192.168.1.0/24 => 00:00:00:00:00:07 0" | simple_switch_CLI
echo "table_add MyIngress.ipv4_lpm MyIngress.forward 192.168.2.0/24 => 00:00:00:00:00:08 1" | simple_switch_CLI
echo "table_add MyIngress.ipv4_lpm MyIngress.forward 192.168.3.10/32 => 00:00:00:00:00:12 2" | simple_switch_CLI
