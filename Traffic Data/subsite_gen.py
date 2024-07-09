def generate_subslice(sub_net_num, slice, switch_site, server_sites):

    server_nics = []
    for i in range(len(server_sites)):
        server_name = "server" + str(i+1)
        server = slice.add_node(name=server_name, site=server_sites[i], cores=4, ram=8, disk=20, image='default_ubuntu_20')
        server_nics.append(server.add_component(model='NIC_Basic').get_interfaces()[0])

    switch = slice.add_node(name=("sub_switch" + str(sub_net_num)), site=switch_site, cores=16, ram=16, disk=40, image='default_ubuntu_20')
    for i in range(len(server_sites)):
        network_name = 'net' + str(i+1)
        switch_nic = switch.add_component(model='NIC_Basic', name=(network_name+'_nic')).get_interfaces()[0]
        slice.add_l2network(name=network_name, interfaces=[server_nics[i], switch_nic])

    return slice
