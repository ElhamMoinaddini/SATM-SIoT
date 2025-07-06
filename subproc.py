import subprocess

# Define the commands for running the scripts
server_command = ['python', 'server.py']
client1_command = ['python', 'client11.py']
client2_command = ['python', 'client3.py']
print("1")
# Start the server in a subprocess
server_process = subprocess.Popen(server_command, stdout=subprocess.PIPE)
print("2")
# Start client 1 in a subprocess
client1_process = subprocess.Popen(client1_command, stdout=subprocess.PIPE)
print("3")
# Start client 2 in a subprocess
client2_process = subprocess.Popen(client2_command, stdout=subprocess.PIPE)
print("4")
# Wait for all subprocesses to complete
server_process.wait()
client1_process.wait()
client2_process.wait()

# Print the output of each subprocess
print("Server output:")
print(server_process.communicate()[0].decode())

print("Client 1 output:")
print(client1_process.communicate()[0].decode())

print("Client 2 output:")
print(client2_process.communicate()[0].decode())

# Print a message when all scripts have finished
print("All scripts have finished executing.")