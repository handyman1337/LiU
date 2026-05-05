from air_lab_interfaces.srv import ExecuteTst
import rclpy
from rclpy.node import Node
import TstML
import ament_index_python
import TstML.Executor
from air_lab3.UndockExecutor import UndockExecutor
from air_lab3.DockExecutor import DockExecutor
from air_lab3.DriveToExecutor import DriveToExecutor
from air_lab3.ExploreExecutor import ExploreExecutor
from air_lab4.RecordSemanticExecutor import RecordSemanticExecutor
from std_srvs.srv import Empty


class MinimalService(Node):

    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(ExecuteTst, 'execute_tst', self.execute_tst_callback)
        self.tst_registry = TstML.TSTNodeModelsRegistry()
        self.tst_registry.loadDirectory(ament_index_python.get_package_prefix("air_tst") +  "/share/air_tst/configs")

        self.callback_group = rclpy.callback_groups.ReentrantCallbackGroup()

           # New services for abort/stop/pause/resume
        self.abort_srv = self.create_service(
            Empty,
            'abort',
            self.callback_abort,
            callback_group=self.callback_group
        )
        self.stop_srv = self.create_service(
            Empty,
            'stop',
            self.callback_stop,
            callback_group=self.callback_group
        )
        self.pause_srv = self.create_service(
            Empty,
            'pause',
            self.callback_pause,
            callback_group=self.callback_group
        )
        self.resume_srv = self.create_service(
            Empty,
            'resume',
            self.callback_resume,
            callback_group=self.callback_group
        )

    def execute_tst_callback(self, request, response):
        
        self.get_logger().info('Incoming request: %s' % (request.tst_file ))
        
        # START
        filename = request.tst_file
        tst_registry = self.tst_registry
        
        # Load TST
        tst_node = TstML.TSTNode.load(filename, tst_registry)

        # Create a registry with node executors
        tst_executor_registry = TstML.Executor.NodeExecutorRegistry()

        # Setup the executors for sequence and concurrent
        tst_executor_registry.registerNodeExecutor(
            tst_registry.model("seq"),
            TstML.Executor.DefaultNodeExecutor.Sequence)
        tst_executor_registry.registerNodeExecutor(
            tst_registry.model("conc"),
            TstML.Executor.DefaultNodeExecutor.Concurrent)
          
        # Also for other tst nodes
        tst_executor_registry.registerNodeExecutor(
            self.tst_registry.model("undock"),
            UndockExecutor)

        tst_executor_registry.registerNodeExecutor(
            self.tst_registry.model("dock"),
            DockExecutor)
        
        tst_executor_registry.registerNodeExecutor(
            self.tst_registry.model("drive-to"),
            DriveToExecutor)
        
        tst_executor_registry.registerNodeExecutor(
            self.tst_registry.model("explore"),
            ExploreExecutor)
        
        tst_executor_registry.registerNodeExecutor(
            self.tst_registry.model("record-semantic"),
            RecordSemanticExecutor)

        # Create an executor using the executors defined
        # in tst_executor_registry
        tst_executor = TstML.Executor.Executor(tst_node,
            tst_executor_registry)

        self.tst_executor = tst_executor
        # Start execution
        tst_executor.start()

        
        # Block until the execution has finished
        status = tst_executor.waitForFinished()

        # Display the result of execution
        # there is a type: TstML.Executor.ExecutionStatus.Type.Aborted
        if status.type() == TstML.Executor.ExecutionStatus.Type.Finished:
            response.success = True
            response.error_message = ""
            
        elif status.type() == TstML.Executor.ExecutionStatus.Type.Failed:
            response.success = False
            response.error_message = status.message()
        else:
            response.success = False
            response.error_message = "Execution aborted or unknown state"


        # END """
        return response
        
    def callback_abort(self, request, response):
        if self.tst_executor:
            self.tst_executor.abort()
            self.get_logger().info("Execution aborted")
        return response

    def callback_stop(self, request, response):
        if self.tst_executor:
            self.tst_executor.stop()
            self.get_logger().info("Execution stopped")
        return response

    def callback_pause(self, request, response):
        if self.tst_executor:
            self.tst_executor.pause()
            self.get_logger().info("Execution paused")
        return response

    def callback_resume(self, request, response):
        if self.tst_executor:
            self.tst_executor.resume()
            self.get_logger().info("Execution resumed")
        return response


def main():
    rclpy.init()

    minimal_service = MinimalService()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(minimal_service)
    executor.spin()

    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()