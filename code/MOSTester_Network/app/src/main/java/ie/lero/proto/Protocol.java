package ie.lero.proto;

import android.util.Log;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.SocketAddress;
import java.nio.ByteBuffer;

public class Protocol
{
    static private final int PAYLOAD_COMMAND = 0;
    static private final int PAYLOAD_AUDIO   = 1;
    static private final int PAYLOAD_TEXT    = 2;

    static public final int COMMAND_CONNECT    = 0;
    static public final int COMMAND_HANDSHAKE  = 1;
    static public final int COMMAND_DISCONNECT = 2;
    static public final int COMMAND_GET_NEXT   = 3;
    static public final int COMMAND_EOF        = 4;
    static public final int COMMAND_EOT        = 5;

    static public abstract class Listener
    {
        protected void onCommand(int command)
        {
            String cmdStr;

            switch (command) {
                case Protocol.COMMAND_CONNECT:
                    cmdStr = "CONNECT";
                    break;

                case Protocol.COMMAND_HANDSHAKE:
                    cmdStr = "HANDSHAKE";
                    break;

                case Protocol.COMMAND_DISCONNECT:
                    cmdStr = "DISCONNECT";
                    break;

                case Protocol.COMMAND_GET_NEXT:
                    cmdStr = "GET_NEXT";
                    break;

                case Protocol.COMMAND_EOF:
                    cmdStr = "END_OF_FILE";
                    break;

                case Protocol.COMMAND_EOT:
                    cmdStr = "END_OF_TEST";
                    break;

                default:
                    cmdStr = "UNKNOWN";
                    break;
            }

            Log.d("Protocol.Listener", "COMMAND: " + cmdStr);
        }

        abstract protected void onAudio(ByteBuffer buffer);
        abstract protected void onText(String text);
    }

    private Listener listener;
    private Network  network;

    private SocketAddress peer;
    private int port;

    public Protocol(int port)
    {
        this.port = port;

        network = new Network(new Network.Listener()
        {
            @Override
            public void onReceive(SocketAddress remoteAddress, ByteBuffer buffer)
            {
                int        type = buffer.getInt();
                ByteBuffer data = buffer.slice();

                switch (type) {
                    case PAYLOAD_COMMAND:
                        int cmd = data.getInt();

                        switch (cmd) {
                            case COMMAND_CONNECT:
                                peer = remoteAddress;
                                handshake();
                                break;
                                
                            case COMMAND_HANDSHAKE:
                                peer = remoteAddress;
                                break;
                                
                            case COMMAND_DISCONNECT:
                                peer = null;
                                listen();
                                break;
                        }

                        listener.onCommand(cmd);
                        break;

                    case PAYLOAD_AUDIO:
                        listener.onAudio(data);
                        break;

                    case PAYLOAD_TEXT:
                        listener.onText(new String(data.array(), data.arrayOffset(), data.limit()));
                        break;
                }
            }
        });
    }

    public boolean isConnected()
    {
        return peer != null;
    }

    public void listen()
    {
        network.listen(port);
    }

    public void setListener(Listener listener)
    {
        this.listener = listener;
    }

    public InetAddress getLocalAddress()
    {
        return network.getLocalAddress();
    }

    public void connect(String address)
    {
        ByteBuffer buffer = ByteBuffer.allocate(8);
        buffer.putInt(PAYLOAD_COMMAND);
        buffer.putInt(COMMAND_CONNECT);
        network.send(new InetSocketAddress(address, port), buffer);
    }

    public void handshake()
    {
        ByteBuffer buffer = ByteBuffer.allocate(8);
        buffer.putInt(PAYLOAD_COMMAND);
        buffer.putInt(COMMAND_HANDSHAKE);
        network.send(peer, buffer);
    }

    public void disconnect()
    {
        ByteBuffer buffer = ByteBuffer.allocate(8);
        buffer.putInt(PAYLOAD_COMMAND);
        buffer.putInt(COMMAND_DISCONNECT);
        network.send(peer, buffer);
    }

    public void sendAudio(ByteBuffer buffer)
    {
        ByteBuffer new_buffer = ByteBuffer.allocate(4 + buffer.limit());
        new_buffer.putInt(PAYLOAD_AUDIO);
        new_buffer.put(buffer);
        network.send(peer, new_buffer);
    }

    public void sendText(String text)
    {
        byte[] bytes = text.getBytes();

        ByteBuffer buffer = ByteBuffer.allocate(4 + bytes.length);
        buffer.putInt(PAYLOAD_TEXT);
        buffer.put(bytes);
        network.send(peer, buffer);
    }

    public void getNext()
    {
        ByteBuffer buffer = ByteBuffer.allocate(8);
        buffer.putInt(PAYLOAD_COMMAND);
        buffer.putInt(COMMAND_GET_NEXT);
        network.send(peer, buffer);
    }
}
