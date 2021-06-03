package ie.lero.proto.tasks;

import android.util.Log;

import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.SocketAddress;
import java.net.SocketTimeoutException;
import java.nio.ByteBuffer;

import ie.lero.proto.Network;

public class ClientTask extends NetTask
{
    public ClientTask(Network network, SocketAddress socketAddress)
    {
        super(network, socketAddress);
    }

    @Override
    protected Void doInBackground(Void... voids)
    {
        try {
            DatagramSocket socket = new DatagramSocket();
            socket.setReuseAddress(true);
            socket.setSoTimeout(10);

            ByteBuffer recvBuffer = ByteBuffer.allocate(65536);
            DatagramPacket packet = new DatagramPacket(recvBuffer.array(), recvBuffer.limit(), socketAddress);

            while (!isCancelled()) {
                while (!sendQueue.isEmpty()) {
                    ByteBuffer sendBuffer = sendQueue.poll();
                    socket.send(new DatagramPacket(sendBuffer.array(), sendBuffer.limit(), packet.getSocketAddress()));
                }

                try {
                    recvBuffer.clear();
                    socket.receive(packet);
                    recvBuffer.limit(packet.getLength());
                    network.getListener().onReceive(packet.getSocketAddress(), recvBuffer);
                } catch (SocketTimeoutException e) { /**/ }
            }

            socket.close();

        } catch (IOException e) {
            Log.e("SERVER_TASK", "doInBackground: ");
        }

        return null;
    }
}
