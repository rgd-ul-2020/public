package ie.lero.proto;

import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.SocketAddress;
import java.nio.ByteBuffer;

import ie.lero.proto.tasks.ClientTask;
import ie.lero.proto.tasks.InetTask;
import ie.lero.proto.tasks.NetTask;
import ie.lero.proto.tasks.ServerTask;

public class Network
{
    static private final String TAG = "NETWORK";

    static abstract public class Listener
    {
        abstract public void onReceive(SocketAddress remoteAddress, ByteBuffer buffer);
    }

    private NetTask netTask;
    private Listener listener;
    
    private InetSocketAddress localAddress;

    Network(Listener listener)
    {
        this.listener = listener;

        InetTask task = new InetTask(this);
        task.executeOnExecutor(SingleThreadExecutor.getInstance("INET"));
    }

    public Listener getListener()
    {
        return listener;
    }

    public InetAddress getLocalAddress()
    {
        if (localAddress != null) {
            return localAddress.getAddress();
        }
        return null;
    }

    public void setLocalAddress(InetSocketAddress localAddress)
    {
        this.localAddress = localAddress;
    }

    public void listen(int port)
    {
        if (netTask != null) {
            netTask.cancel(true);
        }

        netTask = new ServerTask(this, new InetSocketAddress(port));
        netTask.executeOnExecutor(SingleThreadExecutor.getInstance("SOCK"));
    }

    public void close()
    {
        netTask.cancel(true);
    }

    public void send(SocketAddress socketAddress, ByteBuffer buffer)
    {
        if (netTask != null) {
            netTask.cancel(true);
        }

        netTask = new ClientTask(this, socketAddress);
        netTask.executeOnExecutor(SingleThreadExecutor.getInstance("SOCK"));
        netTask.send(buffer);
    }

    public void send(ByteBuffer buffer)
    {
        netTask.send(buffer);
    }
}