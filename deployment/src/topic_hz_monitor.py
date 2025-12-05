#!/usr/bin/env python3

import rospy
import rostopic
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from collections import deque
import time
import yaml

class TopicRateMonitor:
    def __init__(self, topics=None, expected_rates=None, rate_tolerance=0.3, window_size=10):
        rospy.init_node('topic_rate_monitor', anonymous=True)
        assert topics is not None, "You must provide a list of topics to monitor."
        assert expected_rates is not None, "You must provide a list of expected rates for the topics."

        # Parameters
        self.topics_to_monitor = rospy.get_param('~topics', topics)
        self.expected_rates = rospy.get_param('~expected_rates', expected_rates)  # Hz
        self.rate_tolerance = rospy.get_param('~rate_tolerance', rate_tolerance)  # 30% tolerance
        self.window_size = rospy.get_param('~window_size', window_size)  # Number of messages to average
        
        # Storage for timing data
        self.topic_times = {}
        self.subscribers = {}
        
        # Diagnostics publisher
        self.diag_pub = rospy.Publisher('/topics_rate', DiagnosticArray, queue_size=10)
        
        # Subscribe to topics
        for i, topic in enumerate(self.topics_to_monitor):
            self.topic_times[topic] = {
                'times': deque(maxlen=self.window_size),
                'expected_rate': self.expected_rates[i] if i < len(self.expected_rates) else 10.0,
                'last_msg_time': None
            }
            
            # Get topic type
            try:
                topic_type, _, _ = rostopic.get_topic_class(topic, blocking=True)
                if topic_type:
                    self.subscribers[topic] = rospy.Subscriber(
                        topic, 
                        topic_type, 
                        self.topic_callback, 
                        callback_args=topic
                    )
                    rospy.loginfo(f"Monitoring topic: {topic} (expected rate: {self.topic_times[topic]['expected_rate']} Hz)")
                else:
                    rospy.logwarn(f"Could not determine type for topic: {topic}")
            except Exception as e:
                rospy.logerr(f"Failed to subscribe to {topic}: {e}")
        
        # Timer for publishing diagnostics
        self.timer = rospy.Timer(rospy.Duration(1.0), self.publish_diagnostics)
        
    def topic_callback(self, msg, topic_name):
        """Callback for each monitored topic"""
        current_time = time.time()
        
        if self.topic_times[topic_name]['last_msg_time'] is not None:
            dt = current_time - self.topic_times[topic_name]['last_msg_time']
            self.topic_times[topic_name]['times'].append(dt)
        
        self.topic_times[topic_name]['last_msg_time'] = current_time
    
    def calculate_rate(self, topic_name):
        """Calculate current rate for a topic"""
        times = self.topic_times[topic_name]['times']
        
        if len(times) < 2:
            return None
        
        # Average time between messages
        avg_dt = sum(times) / len(times)
        
        if avg_dt > 0:
            return 1.0 / avg_dt
        return None
    
    def publish_diagnostics(self, event):
        """Publish diagnostic messages"""
        msg = DiagnosticArray()
        msg.header.stamp = rospy.Time.now()
        
        for topic_name, data in self.topic_times.items():
            status = DiagnosticStatus()
            status.name = f"Topic Rate: {topic_name}"
            status.hardware_id = topic_name
            
            current_rate = self.calculate_rate(topic_name)
            expected_rate = data['expected_rate']
            
            if current_rate is None:
                status.level = DiagnosticStatus.STALE
                status.message = "Waiting for messages..."
                status.values.append(KeyValue(key="Expected Rate (Hz)", value=str(expected_rate)))
                status.values.append(KeyValue(key="Current Rate (Hz)", value="N/A"))
                rospy.logwarn(f"No messages received yet on topic: {topic_name}")
            else:
                # Check if rate is within tolerance
                min_rate = expected_rate * (1 - self.rate_tolerance)
                max_rate = expected_rate * (1 + self.rate_tolerance)
                
                rate_error = abs(current_rate - expected_rate) / expected_rate
                
                if current_rate < min_rate:
                    status.level = DiagnosticStatus.ERROR
                    status.message = f"{topic_name} Rate too low! {current_rate:.2f} Hz (expected {expected_rate:.2f} Hz)"
                    rospy.logwarn(status.message)
                elif current_rate > max_rate:
                    status.level = DiagnosticStatus.WARN
                    status.message = f"{topic_name} Rate too high! {current_rate:.2f} Hz (expected {expected_rate:.2f} Hz)"
                    rospy.logwarn(status.message)
                else:
                    status.level = DiagnosticStatus.OK
                    status.message = f"Rate OK: {current_rate:.2f} Hz"

                
                status.values.append(KeyValue(key="Expected Rate (Hz)", value=str(expected_rate)))
                status.values.append(KeyValue(key="Current Rate (Hz)", value=f"{current_rate:.2f}"))
                status.values.append(KeyValue(key="Rate Error (%)", value=f"{rate_error*100:.1f}"))
                status.values.append(KeyValue(key="Min Acceptable (Hz)", value=f"{min_rate:.2f}"))
                status.values.append(KeyValue(key="Max Acceptable (Hz)", value=f"{max_rate:.2f}"))
                status.values.append(KeyValue(key="Messages in Window", value=str(len(data['times']))))
            
            # Check for stale messages (no message in 2x expected period)
            # if data['last_msg_time'] is not None:
            #     time_since_last = time.time() - data['last_msg_time']
            #     max_delay = 2.0 / expected_rate if expected_rate > 0 else 1.0
                
            #     if time_since_last > max_delay:
            #         status.level = DiagnosticStatus.STALE
            #         status.message = f"No messages for {time_since_last:.2f}s!"
            #         status.values.append(KeyValue(key="Time Since Last Message (s)", value=f"{time_since_last:.2f}"))
            #         rospy.logwarn(f"Stale topic detected: {topic_name}, last message {time_since_last:.2f}s ago")
            
            msg.status.append(status)
        
        self.diag_pub.publish(msg)
    
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:

        with open('/workspace/src/visualnav-transformer/deployment/config/monitor.yaml', 'r') as f:
            monitor_cfg = yaml.safe_load(f)

        with open('/workspace/src/visualnav-transformer/deployment/config/robot.yaml', 'r') as f:
            robot_cfg = yaml.safe_load(f)

        topics = list(monitor_cfg['topics'].keys())
        expected_rates = list(monitor_cfg['topics'].values())

        cmd_vel_rate = robot_cfg["frame_rate"]
        cmd_vel_topic = robot_cfg["vel_navi_topic"]

        topics.append(cmd_vel_topic)
        expected_rates.append(cmd_vel_rate)

        # Assign /waypoint rate with cmd_vel rate
        expected_rates[topics.index('/waypoint')] = cmd_vel_rate

        rospy.loginfo(f"Monitoring topics: {topics} with expected rates: {expected_rates}")

        monitor = TopicRateMonitor(topics=topics, expected_rates=expected_rates)
        monitor.run()
    except rospy.ROSInterruptException:
        pass
