
import pandapower as pp

def set_switch_states(net):
    """
    1) Create and control bus–bus switches.
    2) Open (take out of service) a line between two bus names (e.g. R13-R14).
    """

    # -------------------------------
    # 1. Bus–Bus Switches
    # -------------------------------
    # Retrieve bus indices using bus names
    bus_R14 = net.bus.index[net.bus["name"] == "Bus R14"][0]
    bus_I1  = net.bus.index[net.bus["name"] == "Bus I1"][0]
    bus_I2  = net.bus.index[net.bus["name"] == "Bus I2"][0]
    bus_c3  = net.bus.index[net.bus["name"] == "Bus C3"][0]
    bus_R9  = net.bus.index[net.bus["name"] == "Bus R9"][0]
    bus_c5  = net.bus.index[net.bus["name"] == "Bus C5"][0]

    # Create bus–bus switches that are controllable
    switch_R14_I1 = pp.create_switch(net, bus=bus_R14, element=bus_I1, et="b",
                                     closed=True, type="CB", name="Switch R14-I1")
    switch_I2_c3  = pp.create_switch(net, bus=bus_I2, element=bus_c3, et="b",
                                     closed=True, type="CB", name="Switch I2-c3")
    switch_R9_c5  = pp.create_switch(net, bus=bus_R9, element=bus_c5, et="b",
                                     closed=True, type="CB", name="Switch R9-c5")

    # Example: open the switch between R14 and I1
    net.switch.at[switch_R14_I1, 'closed'] = True
    # Also open the switch between I2 and c3
    net.switch.at[switch_I2_c3, 'closed'] = False
    # Keep R9-c5 switch closed
    net.switch.at[switch_R9_c5, 'closed'] = False
    
    net.switch.loc[net.switch['name'] == 'S2', 'closed'] = False


    # # -------------------------------
    # # 2. Open a Line (e.g. between R13 and R14)
    # # -------------------------------
    # bus_R13 = net.bus.index[net.bus["name"] == "Bus R13"][0]
    # bus_R14 = net.bus.index[net.bus["name"] == "Bus R14"][0]

    # # Find any line(s) that connect R13 and R14
    # line_candidates = net.line[
    #     ((net.line.from_bus == bus_I1) & (net.line.to_bus == bus_I2)) |
    #     ((net.line.from_bus == bus_I2) & (net.line.to_bus == bus_I1))
    # ]
    # if line_candidates.empty:
    #     print("No line found between Bus I1 and Bus I2.")
    # else:
    #     # Set those line(s) out of service
    #     for idx in line_candidates.index:
    #         net.line.at[idx, 'in_service'] = False
    #         print(f"Line {idx} between I1 and I2 is now out of service.")

    # # Print the switch table and line table to verify
    print("\nSwitch Table:")
    print(net.switch[['name', 'bus', 'element', 'closed']])
    # print("\nLine Table (showing in_service status):")
    # print(net.line[['name', 'from_bus', 'to_bus', 'in_service']])
