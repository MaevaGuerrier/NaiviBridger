#!/usr/bin/env python3
# Only work on NVIDIA Jetson platforms with tegrastats available
# Make sure you specified -v /usr/bin/tegrastats:/usr/bin/tegrastats:ro when launching the docker container

#!/usr/bin/env python3

import rospy
import subprocess
import re
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue

def parse_tegrastats(line):
    """Parse tegrastats output and return averages"""
    data = {}
    
    try:
        # GPU usage
        gpu_match = re.search(r'GR3D_FREQ\s+(\d+)%', line)
        if gpu_match:
            data['gpu'] = float(gpu_match.group(1))
        
        # RAM usage
        ram_match = re.search(r'RAM\s+(\d+)/(\d+)MB', line)
        if ram_match:
            used = int(ram_match.group(1))
            total = int(ram_match.group(2))
            data['memory'] = round((used / total) * 100, 1)
        
        # CPU usage - get all CPU percentages and average them
        cpu_matches = re.findall(r'(\d+)%@\d+', line)
        if cpu_matches:
            cpu_values = [float(x) for x in cpu_matches]
            data['cpu'] = round(sum(cpu_values) / len(cpu_values), 1)
        
    except Exception as e:
        rospy.logwarn(f"Parse error: {e}")
    
    return data

def simple_jetson_monitor():
    rospy.init_node('simple_jetson_monitor', anonymous=True)
    pub = rospy.Publisher('/diagnostics', DiagnosticArray, queue_size=10)
    
    rospy.loginfo("Simple Jetson Monitor started")
    
    # Start tegrastats
    try:
        process = subprocess.Popen(
            ['tegrastats', '--interval', '1000'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1
        )
    except FileNotFoundError:
        rospy.logerr("tegrastats not found!")
        return
    
    try:
        for line in iter(process.stdout.readline, ''):
            if rospy.is_shutdown():
                break
            
            data = parse_tegrastats(line.strip())
            
            if not data:
                continue
            
            # Create diagnostic message
            msg = DiagnosticArray()
            msg.header.stamp = rospy.Time.now()
            
            status = DiagnosticStatus()
            status.name = "Jetson Usage"
            status.hardware_id = "Jetson"
            status.level = DiagnosticStatus.OK
            status.message = "System OK"
            
            # Add simple values
            if 'gpu' in data:
                status.values.append(KeyValue(key="GPU (%)", value=str(data['gpu'])))
            
            if 'cpu' in data:
                status.values.append(KeyValue(key="CPU (%)", value=str(data['cpu'])))
            
            if 'memory' in data:
                status.values.append(KeyValue(key="Memory (%)", value=str(data['memory'])))
            
            msg.status.append(status)
            pub.publish(msg)
            
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down...")
    finally:
        process.terminate()
        process.wait()

if __name__ == '__main__':
    try:
        simple_jetson_monitor()
    except rospy.ROSInterruptException:
        pass
