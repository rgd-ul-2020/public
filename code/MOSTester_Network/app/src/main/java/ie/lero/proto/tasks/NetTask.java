package ie.lero.proto.tasks;

import android.os.AsyncTask;

import java.net.SocketAddress;
import java.nio.ByteBuffer;
import java.util.LinkedList;
import java.util.Queue;

import ie.lero.proto.Network;

public abstract class NetTask extends AsyncTask<Void, Void, Void>
{
    protected Network       network;
    protected SocketAddress socketAddress;

    protected Queue<ByteBuffer> sendQueue;

    public NetTask(Network network, SocketAddress socketAddress)
    {
        this.network       = network;
        this.socketAddress = socketAddress;

        sendQueue = new LinkedList<>();
    }

    public void send(ByteBuffer buffer)
    {
        sendQueue.add(buffer);
    }
}
