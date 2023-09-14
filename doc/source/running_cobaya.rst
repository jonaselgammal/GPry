===================
Running with Cobaya
===================

Although you can input your ``Cobaya`` model into GPry and then use GPry's ``run``
method, if you are used to using Cobaya for inference you might find it easier to
just import GPry as a sampler into ``Cobaya``. This is very simple and straightforward
with just a few commands.

For this simply put ``gpry.CobayaSampler`` into the sampler block of your .yaml file or
info dict. There are a number of options and parameters that are available to choose if
you wish to modify the settings. You can find an overview in CobayaSampler.yaml in the
package files. A more complete set of conventions and instructions will come soon...