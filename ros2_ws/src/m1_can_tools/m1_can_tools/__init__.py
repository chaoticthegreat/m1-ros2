"""M1 CAN tools: Damiao bring-up driver + hardware config/test web node.

Phase-0 of the real-hardware deployment. ``dm_protocol`` is a pure-python,
dependency-free codec for the Damiao (DM-series) CAN protocol; ``transport``
abstracts the physical bus (fake / SocketCAN / serial, the real backends
imported lazily); ``motor_bus`` is the maintenance-mode bus owner; and
``hwconfig_node`` serves the config/test web page on top of it.
"""
